import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
from .convnext import ConvNeXt


class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(ConvNeXt(16, dims=(16, 32), depths=(3, 9,), kernel_sizes=5, patch_size=1),), # 48,48

            nn.Sequential(ConvNeXt(32, dims=(48, 64), depths=(3, 9,), kernel_sizes=5, patch_size=1),), # 24,24

            nn.Sequential(ConvNeXt(64, dims=(96, 128), depths=(3, 9,), kernel_sizes=5, patch_size=1),), # 12,12

            nn.Sequential(ConvNeXt(128, dims=(196, 256), depths=(3, 9,), kernel_sizes=5, patch_size=1),), # 6,6

            nn.Sequential(ConvNeXt(256, dims=(256, 384), depths=(3, 9,), kernel_sizes=5, patch_size=1),), # 3,3

            nn.Sequential(ConvNeXt(384, dims=(384, 512), depths=(3, 9,), kernel_sizes=5, patch_size=1),), # 1,1
            ])

        self.audio_encoder = nn.Sequential(
            ConvNeXt(1, dims=(16, 32, 64, 256), kernel_sizes=5, patch_size=1),
            Conv2d(256, 512, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(2,2), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(2,1), padding=1),
            )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),), # 1,1

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), 
            ConvNeXt(512, dims=(512, 512), depths=(3, 9,), kernel_sizes=5, patch_size=1),
            ), # 3,3

            nn.Sequential(Conv2dTranspose(896, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=2, residual=False),
            ConvNeXt(512, dims=(512, 512), depths=(3, 9,), kernel_sizes=5, patch_size=1),
            ), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvNeXt(384, dims=(384, 384), depths=(3, 9,), kernel_sizes=5, patch_size=1),
            nn.ReflectionPad2d(2),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvNeXt(256, dims=(256, 256), depths=(3, 9,), kernel_sizes=5, patch_size=1),
            nn.ReflectionPad2d(2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=4, residual=False),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvNeXt(128, dims=(128, 128), depths=(3, 9,), kernel_sizes=5, patch_size=1),
            nn.ReflectionPad2d(4),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=4, residual=False),
            nn.ReflectionPad2d(3),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=2, residual=False),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), 
            ), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvNeXt(64, dims=(64, 64), depths=(3, 9,), kernel_sizes=5, patch_size=1),
            nn.ReflectionPad2d(4),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=4, residual=False),
            nn.ReflectionPad2d(4),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=4, residual=False),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            nn.ReflectionPad2d(4),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=4, residual=False),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            nn.ReflectionPad2d(2),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ),# 96,96
            ]) 

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            # print('face_encoder : {}'.format(x.shape))
            feats.append(x)


        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            # print('fd : {}'.format(x.shape))
            try:
                # print('does it break here!')
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            # print('face_decoder : {}'.format(x.shape))
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs



''' Testing the wav2lip network '''
# audio_sequences = torch.rand(5, 1, 80, 16)#indiv_mels
# face_sequences = torch.rand(5, 6, 96, 96) #x

# model = Wav2Lip()
# print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
# print(model(audio_sequences, face_sequences).shape)


        # feats.append(torch.rand(5, 16, 96, 96))
        # feats.append(torch.rand(5, 32, 48, 48))
        # feats.append(torch.rand(5, 64, 24, 24))
        # feats.append(torch.rand(5, 128, 12, 12))
        # feats.append(torch.rand(5, 256, 6, 6))
        # feats.append(torch.rand(5, 384, 3, 3))
        # feats.append(torch.rand(5, 512, 1, 1))
