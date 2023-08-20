import torch
import torch.nn as nn
from torch import Tensor

from other_networks import ConvNeXtBlock, InputStem, OutputStem, DownSample, UpSample, Addition, Concatenation, ReturnInput


# image dimensions must be even
class UNet(nn.Module):
    def __init__(self, arch_dict):
        super().__init__()

        assert isinstance(arch_dict['image_channels'], int)
        assert isinstance(arch_dict['initial_channels'], int)
        assert isinstance(arch_dict['resize_blocks'], list)
        assert isinstance(arch_dict['middle_blocks'], int)
        assert arch_dict['skip_connection'] in ['addition', 'concatenation']

        self.resize_blocks = arch_dict['resize_blocks']
        self.middle_blocks = arch_dict['middle_blocks']

        self.layers = nn.ModuleDict()

        self.layers['input_stem'] = InputStem(in_channels=arch_dict['image_channels'],
                                              out_channels=arch_dict['initial_channels'])

        current_channels = arch_dict['initial_channels']
        for i, down_layer_count in enumerate(self.resize_blocks):
            for j in range(down_layer_count):
                self.layers[f'down_block_{i + 1}-{j + 1}'] = ConvNeXtBlock(current_channels)
            self.layers[f'downsample_{i}'] = DownSample(in_channels=current_channels,
                                                        out_channels=2*current_channels)
            current_channels = 2 * current_channels
        
        for j in range(self.middle_blocks):
            self.layers[f'middle_block_{j + 1}'] = ConvNeXtBlock(current_channels)
        
        for i, up_layer_count in reversed(list(enumerate(self.resize_blocks))):
            # this assert doesn't do anything
            assert current_channels % 2 == 0
            self.layers[f'upsample_{i}'] = UpSample(in_channels=current_channels,
                                                    out_channels=current_channels // 2)
            current_channels = current_channels // 2
            for j in range(up_layer_count):
                if arch_dict['skip_connection'] == 'concatenation' and j == 0:
                    self.layers[f'up_block_{i + 1}-{j + 1}'] = ConvNeXtBlock(2 * current_channels, scale_down=2)
                else:
                    self.layers[f'up_block_{i + 1}-{j + 1}'] = ConvNeXtBlock(current_channels)
        
        self.layers['output_stem'] = OutputStem(in_channels=current_channels,
                                                out_channels=arch_dict['image_channels'])
        
        if arch_dict['skip_connection'] == 'addition':
            self.skip_layer = Addition()
            self.output_skip_layer = Addition()
        elif arch_dict['skip_connection'] == 'concatenation':
            self.skip_layer = Concatenation()
            self.output_skip_layer = ReturnInput()
        else:
            raise ValueError('Invalid skip_connection value!')
    
    def forward(self, x: Tensor) -> Tensor:
        # can dump saved values for skip connections to CPU to save VRAM; will probably be slower
        # device = x.device
        skip_dict = {'input': x}

        x = self.layers['input_stem'](x)
        for i, down_layer_count in enumerate(self.resize_blocks):
            for j in range(down_layer_count):
                x = self.layers[f'down_block_{i + 1}-{j + 1}'](x)
            skip_dict[f'output_layer_{i}'] = x
            x = self.layers[f'downsample_{i}'](x)
        
        for j in range(self.middle_blocks):
            x = self.layers[f'middle_block_{j + 1}'](x)
        
        for i, up_layer_count in reversed(list(enumerate(self.resize_blocks))):
            x = self.layers[f'upsample_{i}'](x)
            # does the concat order matter?
            x = self.skip_layer(x, skip_dict[f'output_layer_{i}'])
            for j in range(up_layer_count):
                x = self.layers[f'up_block_{i + 1}-{j + 1}'](x)
        
        x = self.layers['output_stem'](x)
        # does the concat order matter?
        x = self.output_skip_layer(x, skip_dict['input'])
        return x


def main():
    """
    ARCHITECTURE
    - initial convnext block (image_channels -> initial_channels)
    - 'n' convnext blocks, then downsample; for 'n' in resize_blocks (list)
    - 'n' convnext blocks; for 'n' == middle (int)
    - upsample, then 'n' convnext blocks; for 'n' in resize_blocks (reversed list)
    - final convnext block (initial_channels -> image_channels)
    """

    arch_dict = {
        'image_channels': 3,
        'initial_channels': 64,
        'resize_blocks': [2, 2, 2, 2],
        'middle_blocks': 2,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet = UNet(arch_dict=arch_dict)
    unet.to(device)
    print(unet)

    test_input = torch.randn((8, 3, 128, 128)).to(device)
    test_output = unet.forward(test_input)
    print(test_output.shape)


if __name__ == '__main__':
    main()
