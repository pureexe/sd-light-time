from transformers import pipeline

estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
result = estimator(images="http://images.cocodataset.org/val2017/000000039769.jpg")
print('result')
print(result)