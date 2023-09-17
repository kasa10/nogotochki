import albumentations as album
import argparse
import cv2
import numpy as np
import pickle
import torch


def get_predicted_mask(model, processed_img, device):
    x_tensor = torch.from_numpy(processed_img).to(device).unsqueeze(0)
    pred_mask = model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask,(1,2,0))
    return pred_mask[:,:,1] # 0 - background mask, 1 - nails mask


def prepare_image(image):
    border_v = 0
    border_h = 0
    IMG_COL = 800
    IMG_ROW = 800
    if (IMG_COL/IMG_ROW) >= (image.shape[0]/image.shape[1]):
        border_v = int((((IMG_COL/IMG_ROW)*image.shape[1])-image.shape[0])/2)
    else:
        border_h = int((((IMG_ROW/IMG_COL)*image.shape[0])-image.shape[1])/2)
    image = cv2.copyMakeBorder(image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    image = cv2.resize(image, (IMG_ROW, IMG_COL))
    return image, border_h, border_v


def get_preprocessing(preprocessing_fn=None):
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)


def transform_mask(pred_mask, border_h, border_w, orig_h, orig_w):
    pred_mask_resized = cv2.resize(pred_mask, (orig_h + 2 * border_w, orig_w + 2 * border_h))
    h, w = pred_mask_resized.shape[:2]
    resized_mask = pred_mask_resized[border_w:h-border_w, border_h:w-border_h] * 255
    binary_mask = cv2.threshold(resized_mask, 128, 1, cv2.THRESH_BINARY)[1]
    return np.uint8(binary_mask)


def create_alpha_channel(mask):
    GRADIENT_LAYERS_COUNT = 4
    kernel_3x3 = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask.copy(), kernel_3x3, iterations = 1)
    last_gradient = mask.copy()
    alpha_channel = np.float32(mask.copy())
    for idx in range(0, GRADIENT_LAYERS_COUNT - 1):
        alpha_value = round((GRADIENT_LAYERS_COUNT - 1 - idx) / GRADIENT_LAYERS_COUNT, 2)
        dilation_step = cv2.dilate(last_gradient, kernel_3x3, iterations=1)
        intersection_step = cv2.bitwise_and(dilation_step, cv2.bitwise_not(last_gradient))
        last_gradient = dilation_step.copy()
        alpha_channel += intersection_step * alpha_value

    return alpha_channel

def segment_nails(image_path, model_path, preprocessing_fn_path):
    device = torch.device('cpu')
    model = torch.load(model_path, map_location = device)
    preprocessing_fn = pickle.load(open(preprocessing_fn_path, 'rb'))

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    prepared_img, border_h, border_v = prepare_image(image)
    preprocessing = get_preprocessing(preprocessing_fn)
    preprocessed_image = preprocessing(image=prepared_img)['image']
    pred_mask = get_predicted_mask(model, preprocessed_image, device)
    binary_mask_prototype = transform_mask(pred_mask, border_h, border_v, orig_h, orig_w)
    alpha_channel = create_alpha_channel(binary_mask_prototype)
    binary_mask = np.uint8(cv2.threshold(alpha_channel, 0.01, 1, cv2.THRESH_BINARY)[1])

    binary_mask = cv2.resize(binary_mask, (orig_w, orig_h))
    alpha_channel = cv2.resize(alpha_channel, (orig_w, orig_h))
    return 255 * binary_mask, alpha_channel


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", required = True, help = 'image path for predict')
    args = ap.parse_args()
    model_path = 'best_model_80_v1.pth'
    preprocessing_fn_path = 'preprocessing_fn_80_v1.pkl'
    mask = segment_nails(args.img_path, model_path, preprocessing_fn_path)
    cv2.imwrite('result_mask.jpg', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
