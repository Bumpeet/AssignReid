def pre_process(input):

    image_size = (256, 128)
    transforms = []
    transforms += [T.Resize(image_size)]
    transforms += [T.ToTensor()]
    transforms += [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    preprocess = T.Compose(transforms)

    to_pil = T.ToPILImage()

    image = to_pil(input)
    image = preprocess(image)
    images = image.unsqueeze(0)
    return images


def generate_visuals(RR, imgs, n_images):
    imgs = imgs.ravel()
    np.random.shuffle(imgs)
    RR.extractor.model.eval()
    shutil.rmtree('./output/',  ignore_errors=True)
    os.makedirs('./output/')

    with torch.no_grad():

        for i in tqdm(range(n_images)):
            img = cv2.imread(imgs[i])
            preprocessed = RR.pre_process(img)
            activation = RR.extractor.model(preprocessed, True)

            output = RR.generate_visualization(img, activation, 0.5)

            cv2.imwrite(f'./output/{i}.jpg', output)

    print('---------done generating the activation maps-------------')

    
def run_inference_am(self, imgs):
    processed_imgs = self.pre_process(imgs)
    with torch.no_grad:
        acitvations = self.extractor.model(processed_imgs, True)

    return acitvations

def generate_visualization(self, img: np.ndarray, am: np.ndarray, intensity: float):

    # Set the weight for blending the heatmap and image
    heatmap_weight = intensity  # Adjust this value to control the intensity of the heatmap overlay

    # Overlay the heatmap on the image
    mix = cv2.addWeighted(img, 1 - heatmap_weight, am, heatmap_weight, 0)

    return mix

    # randPerm = torch.randperm(nb*bs)
    # activations = activations[randPerm]
    # imgs = imgs.ravel()[randPerm.numpy()]
    # regex = r"\d+"
    # #grenerating the image and saving it.
    # for i, (am, img) in tqdm(enumerate(zip(ams, imgs)), "Generating the activation maps"):
    #     im = cv2.imread(img)
    #     reg = re.search(regex, img)
    #     name = img[reg.sart(), reg.end()]
    #     output = (1-heatmap_weight)*im + heatmap_weight*am
    #     name = img[img.rfind('/')+1:]
    #     cv2.imwrite(f'./Inference/output/{name}.jpg', output)