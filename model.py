'''
IDEA: Image Food Nutritional Analysis
Takes a photo, analyses the food in the image, guesses what ingredients it's comprised of,
and returns a nutritional score
SDGs 3, 4, 10
'''

from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with your custom model path

# Training the model on our dataset
model.train(
    data="custom_data.yaml",
    epochs=50, # Total training iterations through the dataset
    imgsz=640, # Image size for training
    lr0=0.001, # Initial learning rate
    optimizer="Adam", 
    patience=10, # Number of epochs with no improvement after which training will be stopped
    batch=16, # Number of images processed in one training step
    name="food_nutrition_model"
)
