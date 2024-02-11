# Copyright (c) 2024 Jonathan S. Pollack (https://github.com/JPPhoto)
# Original octree implementation via https://github.com/delimitry/octree_color_quantizer/
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


class Color(object):
    def __init__(self, red=0, green=0, blue=0, alpha=None):
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha


class OctreeNode(object):
    def __init__(self, level, parent):
        self.color = Color(0, 0, 0)
        self.pixel_count = 0
        self.palette_index = 0
        self.children = [None for _ in range(8)]
        if level < OctreeQuantizer.MAX_DEPTH - 1:
            parent.add_level_node(level, self)

    def is_leaf(self):
        return self.pixel_count > 0

    def get_leaf_nodes(self):
        leaf_nodes = []
        for i in range(8):
            node = self.children[i]
            if node:
                if node.is_leaf():
                    leaf_nodes.append(node)
                else:
                    leaf_nodes.extend(node.get_leaf_nodes())
        return leaf_nodes

    def get_nodes_pixel_count(self):
        sum_count = self.pixel_count
        for i in range(8):
            node = self.children[i]
            if node:
                sum_count += node.pixel_count
        return sum_count

    def add_color(self, color, level, parent):
        if level >= OctreeQuantizer.MAX_DEPTH:
            self.color.red += color.red
            self.color.green += color.green
            self.color.blue += color.blue
            self.pixel_count += 1
            return
        index = self.get_color_index_for_level(color, level)
        if not self.children[index]:
            self.children[index] = OctreeNode(level, parent)
        self.children[index].add_color(color, level + 1, parent)

    def get_palette_index(self, color, level):
        if self.is_leaf():
            return self.palette_index
        index = self.get_color_index_for_level(color, level)
        if self.children[index]:
            return self.children[index].get_palette_index(color, level + 1)
        else:
            # get palette index for a first found child node
            for i in range(8):
                if self.children[i]:
                    return self.children[i].get_palette_index(color, level + 1)

    def remove_leaves(self):
        result = 0
        for i in range(8):
            node = self.children[i]
            if node:
                self.color.red += node.color.red
                self.color.green += node.color.green
                self.color.blue += node.color.blue
                self.pixel_count += node.pixel_count
                result += 1
        return result - 1

    def get_color_index_for_level(self, color, level):
        index = 0
        mask = 0x80 >> level
        if color.red & mask:
            index |= 4
        if color.green & mask:
            index |= 2
        if color.blue & mask:
            index |= 1
        return index

    def get_color(self):
        return Color(
            self.color.red // self.pixel_count,
            self.color.green // self.pixel_count,
            self.color.blue // self.pixel_count,
        )


class OctreeQuantizer(object):
    MAX_DEPTH = 8

    def __init__(self, max_depth=8):
        OctreeQuantizer.MAX_DEPTH = max_depth
        self.levels = {i: [] for i in range(OctreeQuantizer.MAX_DEPTH)}
        self.root = OctreeNode(0, self)

    def get_leaves(self):
        return list(self.root.get_leaf_nodes())

    def add_level_node(self, level, node):
        self.levels[level].append(node)

    def add_color(self, color):
        # passes self value as `parent` to save nodes to levels dict
        self.root.add_color(color, 0, self)

    def make_palette(self, color_count):
        palette = []
        palette_index = 0
        leaf_count = len(self.get_leaves())
        # reduce nodes
        # up to 8 leaves can be reduced here and the palette will have
        # only 248 colors (in worst case) instead of expected 256 colors
        for level in range(OctreeQuantizer.MAX_DEPTH - 1, -1, -1):
            if self.levels[level]:
                for node in self.levels[level]:
                    leaf_count -= node.remove_leaves()
                    if leaf_count <= color_count:
                        break
                if leaf_count <= color_count:
                    break
                self.levels[level] = []
        # build palette
        for node in self.get_leaves():
            if palette_index >= color_count:
                break
            if node.is_leaf():
                palette.append(node.get_color())
            node.palette_index = palette_index
            palette_index += 1
        return palette

    def get_palette_index(self, color):
        return self.root.get_palette_index(color, 0)


@invocation("octree_quantizer", title="Octree Quantizer", tags=["octree quantizer", "image"], version="1.0.0")
class OctreeQuantizerInvocation(BaseInvocation, WithMetadata):
    """Quantizes an image to the desired number of colors"""

    image: ImageField = InputField(description="The image to quantize")
    final_colors: int = InputField(gt=0, description="The final number of colors in the palette", default=16)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        width, height = image.size
        mode = image.mode

        alpha_channel = image.getchannel("A") if mode == "RGBA" else None

        image = image.convert("RGB")

        octree = OctreeQuantizer()

        for y in range(height):
            for x in range(width):
                c = image.getpixel((x, y))
                octree.add_color(Color(c[0], c[1], c[2]))

        # 256 colors for 8 bits per pixel output image
        palette = octree.make_palette(self.final_colors)

        # Quantize
        for y in range(height):
            for x in range(width):
                c = image.getpixel((x, y))
                index = octree.get_palette_index(Color(c[0], c[1], c[2]))
                color = palette[index]
                image.putpixel((x, y), (color.red, color.green, color.blue))

        # Make the image RGBA if we had a source alpha channel
        if alpha_channel is not None:
            image.putalpha(alpha_channel)

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image.width,
            height=image.height,
        )
