# the backbone code is in /facerecognition

global_config = utils.load_global_para(
    './facerecognition/data/global_config.json')
global_para = utils.load_global_para('./facerecognition/data/global_para.json')
configs = {'config': global_config['fd_config'],
           'para': global_para['fd_para']}


def detect_by_mtcnn(img_list, assert_one=True):
    mtcnn_model = Mtcnn(config=configs['config'], para=configs['para'])
    align_img_list = []
    for img in img_list:
        img, img_fd, ratio = read_img_from_opencv(img, configs)
        bboxes, landmarks = mtcnn_model.detect_face(img_fd)
        margin_bboxes = margin_operation_opencv(
            bboxes, img_fd.shape[0:2], configs)

        bboxes_ori = margin_bboxes * ratio
        landmarks_ori = landmarks * ratio

        batch_imgs = face_utils.align_multi(img, landmarks_ori.transpose())
        if len(batch_imgs) > 0:
            if assert_one and len(batch_imgs) > 1:
                continue
            if assert_one:
                cropped_img = Image.fromarray(
                    batch_imgs[0].astype('uint8')).convert('L')
                align_img_list.append(cropped_img)
            else:
                continue
        else:
            continue

    return align_img_list


def margin_operation_opencv(bounding_boxes, img_size, configs):
    margin_bboxes = []

    if configs['para']['margin'] >= 1:
        for bbox in bounding_boxes:
            bbox[0] = np.maximum(bbox[0] - configs['para']['margin'], 0)
            bbox[1] = np.maximum(bbox[1] - configs['para']['margin'], 0)
            bbox[2] = np.minimum(
                bbox[2] + configs['para']['margin'], img_size[1])
            bbox[3] = np.minimum(
                bbox[3] + configs['para']['margin'], img_size[0])
            margin_bboxes.append(bbox)
    else:
        for bbox in bounding_boxes:
            margin_w = configs['para']['margin'] * abs(bbox[0] - bbox[2])
            margin_h = configs['para']['margin'] * abs(bbox[1] - bbox[3])
            box_margin = bbox.copy()
            box_margin[0] = np.maximum(bbox[0]-margin_w, 0)
            box_margin[1] = np.maximum(bbox[1]-margin_h, 0)
            box_margin[2] = np.minimum(bbox[2]+margin_w, img_size[1])
            box_margin[3] = np.minimum(bbox[3]+margin_h, img_size[0])
            margin_bboxes.append(box_margin)

    return np.array(margin_bboxes)


def read_img_from_opencv(img_path, configs):
    try:
        img = cv2.imread(img_path)
        if img.ndim < 2:
            raise Exception('invalid dim of input img!')
        if img.ndim == 2:
            img = utils.to_rgb(img)
        elif img.ndim == 3:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

        detect_w = configs['config']['detect_w']
        if img.shape[1] > detect_w:
            detect_h = int(img.shape[0] * detect_w / img.shape[1])
            img_fd = cv2.resize(img, (detect_w, detect_h),
                                interpolation=cv2.INTER_LINEAR)
            ratio = img.shape[1] / detect_w

            return img, img_fd, ratio
        else:
            img_fd = img.copy()
            ratio = 1

            return img, img_fd, ratio
    except Exception as e:
        print(e)
