from ultralytics import YOLO

model = YOLO(r'E:\ProjectsTest\Yolo\Yolofk\FootballAI\models\best.pt')


results = model.predict(r"E:\ProjectsTest\Yolo\Yolofk\FootballAI\football_analysis\input_videos\100.mp4",save = True)

print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)