# from struction import run
# import os
# import time
# from src.detection.ctpn.text_detect import ctpn
# # init ctpn model
# ctpn_model = ctpn()


# path = "./test_invoice"
# im = os.listdir(path)
# images = [os.path.join(path, _) for _ in im]
# start = time.time()
# df = run(images, ctpn_model)
# end = time.time()
# df.to_csv("result.csv", index=None)
# print(end - start)


