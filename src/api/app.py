from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import cv2
from fliqe import OnlineFLIQE
from confluent_kafka import Producer
import json

conf = {
    'bootstrap.servers': 'archimedes-dev.local:9093',
    'security.protocol': 'ssl',
    'ssl.key.location': 'kafka_certificates/signed.key',
    'ssl.certificate.location': 'kafka_certificates/signed.pem',
    'ssl.ca.location': 'kafka_certificates/trusted_authority.cert'
}
producer = Producer(conf)

class ProcessorConfig(BaseModel):
    video_uri: str = "rtsp://aiqready:aiqready123!@192.168.112.39/axis-media/media.amp?videocodec=h264"
    kafka_topic: str = "image-quality-data"
    smoothing_window: int = 150


class RtspToKafkaProcessor:
    def __init__(self, video_uri, kafka_topic, smoothing_window):
        self.video_uri = video_uri
        self.kafka_topic = kafka_topic
        self.running_task = None
        self.rtsp = None
        self.cap = None
        self.fliqe = OnlineFLIQE(quality_model_path='models/encoder_with_binary_head.pth', smoothing_window=smoothing_window)

    async def start(self):
        print("Starting RTSP → Kafka processing pipeline")
        if self.running_task:
            raise HTTPException(status_code=400, detail="Task is already running.")
        
        self.cap = cv2.VideoCapture(self.video_uri)
        if not self.cap.isOpened():
            print("Cannot open RTSP stream")
            exit()
        self.running_task = asyncio.create_task(self.online_fliqe_task())    

    async def stop(self):
        print("Stopping RTSP → Kafka processing pipeline")
        if not self.running_task:
            raise HTTPException(status_code=400, detail="No running task.")

        if self.running_task:
            self.running_task.cancel()
            self.running_task = None
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Stopping completed.")

    async def online_fliqe_task(self):
        print("Starting online FLIQE quality estimation task")
        self.fliqe.create_session(self.video_uri)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise HTTPException(status_code=500, detail=f"Cannot open RTSP stream: {self.video_uri}")

            _ = self.fliqe.estimate_smoothed_quality(frame, session_id=self.video_uri)
            smoothed_score = self.fliqe.get_smoothed_quality(self.video_uri)
            raw_score = self.fliqe.get_raw_quality(self.video_uri)
            print(f"FLIQE Smoothed Score: {smoothed_score}, Raw Score: {raw_score}")
            message_payload = {
                "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                "smoothing_window": self.fliqe.smoothing_window, 
                "raw_image_quality": raw_score, 
                "smoothed_image_quality": smoothed_score 
            }
            message_json = json.dumps(message_payload)
            producer.produce(self.kafka_topic, message_json.encode('utf-8'))
            producer.flush()
            print("Published FLIQE scores to Kafka topic.")
            await asyncio.sleep(1)

processor: RtspToKafkaProcessor = None

app = FastAPI(
    title="FLIR Image Quality Estimator (FLIQE) API",
    description="This is an API for broadcasting FLIR Image Quality Scores.",
    version="1.0.0",
)


@app.post("/start")
async def start_broadcast(config: ProcessorConfig = ProcessorConfig()):
    global processor
    
    if processor and processor.running_task:
        raise HTTPException(status_code=400, detail="Processor is already running.")
    
    processor = RtspToKafkaProcessor(
        video_uri=config.video_uri,
        kafka_topic=config.kafka_topic,
        smoothing_window=config.smoothing_window
    )
    
    await processor.start()
    return {
        "status": "RTSP to Kafka processing started.",
        "config": config.model_dump()
    }


@app.post("/stop")
async def stop_broadcast():
    global processor
    
    if not processor:
        raise HTTPException(status_code=400, detail="No processor instance found.")
    
    await processor.stop()
    processor = None
    return {"status": "RTSP to Kafka processing stopped."}


@app.get("/status")
async def get_status():
    global processor
    
    if not processor:
        return {"status": "stopped", "processor": None}
    
    is_running = processor.running_task is not None and not processor.running_task.done()
    
    return {
        "status": "running" if is_running else "stopped",
        "processor": {
            "video_uri": processor.video_uri,
            "kafka_topic": processor.kafka_topic,
            "smoothing_window": processor.fliqe.smoothing_window if processor.fliqe else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)