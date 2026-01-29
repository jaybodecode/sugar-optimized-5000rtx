import torch
import lpips

print('Loading LPIPS...')
lpips_fn = lpips.LPIPS(net='vgg').cuda()
print('LPIPS loaded')

print('Compiling LPIPS...')
lpips_fn = torch.compile(lpips_fn)
print('LPIPS compiled')

print('Testing with dummy data...')
img1 = torch.randn(1, 3, 256, 256).cuda()
img2 = torch.randn(1, 3, 256, 256).cuda()
loss = lpips_fn(img1, img2)
print(f'LPIPS loss: {loss.item()}')
print('Success!')
