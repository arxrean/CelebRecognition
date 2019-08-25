def readImage(img_path):
    try:
        return Image.open(img_path).convert('RGB')
    except:
        print('error open img:{}'.format(img_path))
        return None


def filter_imgs(img_path_list):
    img_list = list(map(lambda x: readImage(x), img_path_list))

    p2h = {}
    for i, img in enumerate(img_list):
        if img is not None:
            p2h[img_path_list[i]] = phash(img)

    h2p = {}
    for p, h in p2h.items():
        if h not in h2p:
            h2p[h] = []
        if p not in h2p[h]:
            h2p[h].append(p)

    hs = list(h2p.keys())
    h2h = {}
    for i, h1 in enumerate(hs):
        for h2 in hs[:i]:
            if h1-h2 <= 5:
                h2h[h1] = h2

    valid_img_path_list = []
    for i, img in enumerate(img_list):
        if p2h[img_path_list[i]] not in h2h and img is not None:
            valid_img_path_list.append(img_path_list[i])

    return valid_img_path_list