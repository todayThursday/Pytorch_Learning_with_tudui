from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")
img=Image.open('bird2.jpg')
print(img)

trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("to tensor",img_tensor)


#  normalize
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([1,3,2],[3,2,1])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize",img_norm,2)

# resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
print(img_resize)
img_resize=trans_totensor(img_resize)
writer.add_image("resize",img_resize)

# compose

trans_resize2=transforms.Resize((256,256))
trans_compose=transforms.Compose([trans_resize2,trans_totensor])
img_compose=trans_compose(img)
writer.add_image('compose',img_compose,1)

# randomcrop
trans_randomcrop=transforms.RandomCrop((512,256))
for i in range(10):
    img_randomcrop=trans_randomcrop(img_tensor)
    writer.add_image('crop',img_randomcrop,i)
writer.close()
