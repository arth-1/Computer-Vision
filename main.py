from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video(r'E:\ProjectsTest\Yolo\Yolofk\FootballAI\football_analysis\input_videos\100.mp4')

    # Pass model path when initializing the Tracker
    tracker = Tracker(r'E:\ProjectsTest\Yolo\Yolofk\FootballAI\models\best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    save_video(video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
