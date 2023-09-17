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
    binary_mask = cv2.threshold(resized_mask, 128, 255, cv2.THRESH_BINARY)[1]
    return np.uint8(binary_mask)
    
def segment_nails(image_path, model_path, preprocessing_fn_path):
    device = torch.device('cpu')
    model = torch.load(model_path, map_location = device)
    preprocessing_fn = pickle.load(open(preprocessing_fn_path, 'rb'))

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    prepared_img, border_h, border_v = prepare_image(image)
    # cv2.imwrite('prepared.jpg', cv2.cvtColor(prepared_img, cv2.COLOR_RGB2BGR))
    preprocessing = get_preprocessing(preprocessing_fn)
    preprocessed_image = preprocessing(image=prepared_img)['image']
    # cv2.imwrite('preprocessed.jpg', cv2.cvtColor(preprocessed_image.transpose([1, 2, 0]), cv2.COLOR_RGB2BGR) * 128)
    pred_mask = get_predicted_mask(model, preprocessed_image, device)
    # cv2.imwrite('pred_mask.jpg', cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR) * 255)
    transformed_mask = transform_mask(pred_mask, border_h, border_v, orig_h, orig_w)
    
    # orig_h, orig_w = transformed_mask.shape[:2]
    # print('transformed_mask orig_h, orig_w', orig_h, orig_w)
    # cv2.imwrite('transformed_mask.jpg', cv2.cvtColor(transformed_mask, cv2.COLOR_RGB2BGR))
    
    # masked_nails = image.copy()
    # for i in range(3):
    #     print(transformed_mask)
    #     masked_nails[:,:,i] = cv2.bitwise_and(masked_nails[:,:,i], masked_nails[:,:,i], mask=transformed_mask)
    # cv2.imwrite('masked_nails.jpg', cv2.cvtColor(masked_nails, cv2.COLOR_RGB2BGR))
    
    return transformed_mask


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", required = True, help = 'image path for predict')
    args = ap.parse_args()
    model_path = 'best_model_80_v1.pth'
    preprocessing_fn_path = 'preprocessing_fn_80_v1.pkl'
    mask = segment_nails(args.img_path, model_path, preprocessing_fn_path)
    cv2.imwrite('result_mask.jpg', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
