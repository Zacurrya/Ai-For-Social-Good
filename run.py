
import os
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'resnet50_food101.pth')
CLASS_TXT_PATH = os.path.join(BASE_DIR, 'food101_classes.txt')

# === Nutrition Data ===
nutrition_data = {
    'apple_pie': '- Food: Apple Pie\n- Calories: 320 per slice\n- Health Score: 4/10\n- Summary: Delicious but high in sugar and fat.',
    'baby_back_ribs': '- Food: Baby Back Ribs\n- Calories: 500 per serving\n- Health Score: 3/10\n- Summary: High in protein but also saturated fats and sodium.',
    'baklava': '- Food: Baklava\n- Calories: 290 per piece\n- Health Score: 4/10\n- Summary: Rich in sugar and fat, best enjoyed occasionally.',
    'beef_carpaccio': '- Food: Beef Carpaccio\n- Calories: 150 per serving\n- Health Score: 7/10\n- Summary: Lean protein with minimal fat, healthy if prepared safely.',
    'beef_tartare': '- Food: Beef Tartare\n- Calories: 180 per serving\n- Health Score: 6/10\n- Summary: Good protein source but raw consumption carries health risks.',
    'beet_salad': '- Food: Beet Salad\n- Calories: 120 per serving\n- Health Score: 8/10\n- Summary: Low-calorie and nutrient-rich.',
    'beignets': '- Food: Beignets\n- Calories: 250 per piece\n- Health Score: 3/10\n- Summary: Deep-fried and sugary, a rare treat.',
    'bibimbap': '- Food: Bibimbap\n- Calories: 500 per bowl\n- Health Score: 7/10\n- Summary: Balanced mix of rice, vegetables, and protein.',
    'bread_pudding': '- Food: Bread Pudding\n- Calories: 350 per serving\n- Health Score: 4/10\n- Summary: Tasty but high in sugar and carbs.',
    'breakfast_burrito': '- Food: Breakfast Burrito\n- Calories: 600 per burrito\n- Health Score: 5/10\n- Summary: Filling but can be high in fat and sodium.',
    'bruschetta': '- Food: Bruschetta\n- Calories: 90 per piece\n- Health Score: 6/10\n- Summary: Light appetizer with tomatoes and herbs.',
    'caesar_salad': '- Food: Caesar Salad\n- Calories: 280 per serving\n- Health Score: 5/10\n- Summary: Nutritious greens but high-calorie dressing.',
    'cannoli': '- Food: Cannoli\n- Calories: 300 per piece\n- Health Score: 3/10\n- Summary: Sweet Italian pastry high in sugar and fat.',
    'caprese_salad': '- Food: Caprese Salad\n- Calories: 200 per serving\n- Health Score: 7/10\n- Summary: Fresh and light with healthy fats from mozzarella.',
    'carrot_cake': '- Food: Carrot Cake\n- Calories: 400 per slice\n- Health Score: 4/10\n- Summary: Contains vegetables but high in sugar and calories.',
    'ceviche': '- Food: Ceviche\n- Calories: 140 per serving\n- Health Score: 8/10\n- Summary: Fresh seafood dish low in calories and high in protein.',
    'cheesecake': '- Food: Cheesecake\n- Calories: 450 per slice\n- Health Score: 3/10\n- Summary: Rich dessert very high in sugar and fat.',
    'cheese_plate': '- Food: Cheese Plate\n- Calories: 350 per serving\n- Health Score: 5/10\n- Summary: Good protein and calcium but high in saturated fat.',
    'chicken_curry': '- Food: Chicken Curry\n- Calories: 420 per serving\n- Health Score: 6/10\n- Summary: Protein-rich with beneficial spices but can be high in fat.',
    'chicken_quesadilla': '- Food: Chicken Quesadilla\n- Calories: 480 per serving\n- Health Score: 5/10\n- Summary: Good protein source but high in calories and fat.',
    'chicken_wings': '- Food: Chicken Wings\n- Calories: 320 per serving\n- Health Score: 4/10\n- Summary: High protein but often fried and high in sodium.',
    'chocolate_cake': '- Food: Chocolate Cake\n- Calories: 380 per slice\n- Health Score: 3/10\n- Summary: Indulgent dessert high in sugar and calories.',
    'chocolate_mousse': '- Food: Chocolate Mousse\n- Calories: 280 per serving\n- Health Score: 4/10\n- Summary: Rich dessert high in sugar and fat.',
    'churros': '- Food: Churros\n- Calories: 200 per piece\n- Health Score: 3/10\n- Summary: Fried dough coated in sugar, occasional treat.',
    'clam_chowder': '- Food: Clam Chowder\n- Calories: 320 per cup\n- Health Score: 5/10\n- Summary: Good protein from clams but high in cream and calories.',
    'club_sandwich': '- Food: Club Sandwich\n- Calories: 540 per sandwich\n- Health Score: 5/10\n- Summary: Balanced meal but high in calories and sodium.',
    'crab_cakes': '- Food: Crab Cakes\n- Calories: 250 per cake\n- Health Score: 6/10\n- Summary: Good seafood protein but often fried.',
    'creme_brulee': '- Food: Creme Brulee\n- Calories: 340 per serving\n- Health Score: 3/10\n- Summary: Rich custard dessert high in sugar and cream.',
    'croque_madame': '- Food: Croque Madame\n- Calories: 520 per serving\n- Health Score: 4/10\n- Summary: Hearty French dish high in calories and fat.',
    'cup_cakes': '- Food: Cupcakes\n- Calories: 240 per cupcake\n- Health Score: 3/10\n- Summary: Sweet treat high in sugar and refined flour.',
    'deviled_eggs': '- Food: Deviled Eggs\n- Calories: 80 per egg\n- Health Score: 6/10\n- Summary: Good protein source but high in cholesterol.',
    'donuts': '- Food: Donuts\n- Calories: 260 per donut\n- Health Score: 2/10\n- Summary: Fried and glazed, very high in sugar and fat.',
    'dumplings': '- Food: Dumplings\n- Calories: 40 per dumpling\n- Health Score: 6/10\n- Summary: Moderate calories, good if steamed rather than fried.',
    'edamame': '- Food: Edamame\n- Calories: 95 per half cup\n- Health Score: 9/10\n- Summary: Excellent plant protein source, low calorie and nutritious.',
    'eggs_benedict': '- Food: Eggs Benedict\n- Calories: 440 per serving\n- Health Score: 5/10\n- Summary: Good protein but high in calories from hollandaise sauce.',
    'escargots': '- Food: Escargots\n- Calories: 180 per serving\n- Health Score: 6/10\n- Summary: Low-calorie protein but high in butter and garlic.',
    'falafel': '- Food: Falafel\n- Calories: 60 per ball\n- Health Score: 7/10\n- Summary: Good plant protein source, especially when baked.',
    'filet_mignon': '- Food: Filet Mignon\n- Calories: 280 per 6 oz\n- Health Score: 7/10\n- Summary: Lean cut of beef, high in protein and iron.',
    'fish_and_chips': '- Food: Fish and Chips\n- Calories: 720 per serving\n- Health Score: 4/10\n- Summary: Good protein from fish but high calories from frying.',
    'foie_gras': '- Food: Foie Gras\n- Calories: 130 per ounce\n- Health Score: 3/10\n- Summary: Very high in fat and cholesterol, luxury item.',
    'french_fries': '- Food: French Fries\n- Calories: 320 per serving\n- Health Score: 3/10\n- Summary: High in calories, fat, and sodium from frying.',
    'french_onion_soup': '- Food: French Onion Soup\n- Calories: 380 per bowl\n- Health Score: 5/10\n- Summary: Flavorful but high in calories from cheese and broth.',
    'french_toast': '- Food: French Toast\n- Calories: 280 per slice\n- Health Score: 4/10\n- Summary: Breakfast treat high in sugar and refined carbs.',
    'fried_calamari': '- Food: Fried Calamari\n- Calories: 300 per serving\n- Health Score: 4/10\n- Summary: Good protein but high calories from frying.',
    'fried_rice': '- Food: Fried Rice\n- Calories: 350 per cup\n- Health Score: 5/10\n- Summary: Balanced meal but high in sodium and oil.',
    'frozen_yogurt': '- Food: Frozen Yogurt\n- Calories: 120 per half cup\n- Health Score: 6/10\n- Summary: Lower fat alternative to ice cream with probiotics.',
    'garlic_bread': '- Food: Garlic Bread\n- Calories: 180 per slice\n- Health Score: 4/10\n- Summary: Tasty side dish but high in refined carbs and butter.',
    'gnocchi': '- Food: Gnocchi\n- Calories: 250 per cup\n- Health Score: 5/10\n- Summary: Potato dumplings, moderate calories but refined carbs.',
    'greek_salad': '- Food: Greek Salad\n- Calories: 220 per serving\n- Health Score: 8/10\n- Summary: Fresh vegetables with healthy fats from olives and feta.',
    'grilled_cheese_sandwich': '- Food: Grilled Cheese Sandwich\n- Calories: 400 per sandwich\n- Health Score: 4/10\n- Summary: Comfort food high in calories and saturated fat.',
    'grilled_salmon': '- Food: Grilled Salmon\n- Calories: 280 per 6 oz\n- Health Score: 9/10\n- Summary: Excellent omega-3 source, high protein and heart-healthy.',
    'guacamole': '- Food: Guacamole\n- Calories: 150 per quarter cup\n- Health Score: 8/10\n- Summary: Healthy fats from avocado, nutritious and satisfying.',
    'gyoza': '- Food: Gyoza\n- Calories: 40 per dumpling\n- Health Score: 6/10\n- Summary: Japanese dumplings, moderate calories when steamed.',
    'hamburger': '- Food: Hamburger\n- Calories: 540 per burger\n- Health Score: 4/10\n- Summary: Good protein but high in calories and saturated fat.',
    'hot_and_sour_soup': '- Food: Hot and Sour Soup\n- Calories: 90 per cup\n- Health Score: 7/10\n- Summary: Low-calorie soup with vegetables and spices.',
    'hot_dog': '- Food: Hot Dog\n- Calories: 280 per hot dog\n- Health Score: 3/10\n- Summary: Processed meat high in sodium and preservatives.',
    'huevos_rancheros': '- Food: Huevos Rancheros\n- Calories: 380 per serving\n- Health Score: 6/10\n- Summary: Protein-rich breakfast with vegetables and beans.',
    'hummus': '- Food: Hummus\n- Calories: 100 per quarter cup\n- Health Score: 8/10\n- Summary: Plant protein and fiber from chickpeas, very nutritious.',
    'ice_cream': '- Food: Ice Cream\n- Calories: 200 per half cup\n- Health Score: 3/10\n- Summary: Frozen dessert high in sugar and saturated fat.',
    'lasagna': '- Food: Lasagna\n- Calories: 450 per serving\n- Health Score: 5/10\n- Summary: Hearty dish with protein but high in calories and fat.',
    'lobster_bisque': '- Food: Lobster Bisque\n- Calories: 280 per cup\n- Health Score: 5/10\n- Summary: Rich seafood soup high in cream and calories.',
    'lobster_roll_sandwich': '- Food: Lobster Roll Sandwich\n- Calories: 380 per roll\n- Health Score: 6/10\n- Summary: Good seafood protein but high in mayonnaise.',
    'macaroni_and_cheese': '- Food: Macaroni and Cheese\n- Calories: 320 per cup\n- Health Score: 4/10\n- Summary: Comfort food high in refined carbs and saturated fat.',
    'macarons': '- Food: Macarons\n- Calories: 80 per macaron\n- Health Score: 4/10\n- Summary: Delicate French cookie high in sugar and almond flour.',
    'miso_soup': '- Food: Miso Soup\n- Calories: 40 per cup\n- Health Score: 8/10\n- Summary: Low-calorie soup with probiotics and minimal fat.',
    'mussels': '- Food: Mussels\n- Calories: 150 per serving\n- Health Score: 8/10\n- Summary: Excellent lean protein source with vitamins and minerals.',
    'nachos': '- Food: Nachos\n- Calories: 480 per serving\n- Health Score: 3/10\n- Summary: High in calories, fat, and sodium from cheese and chips.',
    'omelette': '- Food: Omelette\n- Calories: 220 per omelette\n- Health Score: 7/10\n- Summary: High protein breakfast, nutritious with vegetables.',
    'onion_rings': '- Food: Onion Rings\n- Calories: 280 per serving\n- Health Score: 3/10\n- Summary: Battered and fried, high in calories and fat.',
    'oysters': '- Food: Oysters\n- Calories: 50 per oyster\n- Health Score: 8/10\n- Summary: Low-calorie shellfish high in zinc and protein.',
    'pad_thai': '- Food: Pad Thai\n- Calories: 400 per serving\n- Health Score: 6/10\n- Summary: Balanced noodle dish but can be high in sodium.',
    'paella': '- Food: Paella\n- Calories: 380 per serving\n- Health Score: 7/10\n- Summary: Spanish rice dish with seafood and vegetables.',
    'pancakes': '- Food: Pancakes\n- Calories: 200 per pancake\n- Health Score: 4/10\n- Summary: Breakfast treat high in refined flour and sugar.',
    'panna_cotta': '- Food: Panna Cotta\n- Calories: 260 per serving\n- Health Score: 4/10\n- Summary: Italian dessert high in cream and sugar.',
    'peking_duck': '- Food: Peking Duck\n- Calories: 340 per serving\n- Health Score: 5/10\n- Summary: Flavorful dish but high in fat from skin.',
    'pho': '- Food: Pho\n- Calories: 350 per bowl\n- Health Score: 7/10\n- Summary: Vietnamese soup with protein and vegetables, relatively healthy.',
    'pizza': '- Food: Pizza\n- Calories: 280 per slice\n- Health Score: 4/10\n- Summary: Popular food but high in calories and refined carbs.',
    'pork_chop': '- Food: Pork Chop\n- Calories: 320 per 6 oz\n- Health Score: 6/10\n- Summary: Good protein source but can be high in saturated fat.',
    'poutine': '- Food: Poutine\n- Calories: 520 per serving\n- Health Score: 2/10\n- Summary: Canadian dish very high in calories and fat.',
    'prime_rib': '- Food: Prime Rib\n- Calories: 420 per 6 oz\n- Health Score: 5/10\n- Summary: High protein but also high in saturated fat.',
    'pulled_pork_sandwich': '- Food: Pulled Pork Sandwich\n- Calories: 480 per sandwich\n- Health Score: 5/10\n- Summary: Good protein but high in calories and sodium.',
    'ramen': '- Food: Ramen\n- Calories: 380 per bowl\n- Health Score: 5/10\n- Summary: Satisfying noodle soup but can be high in sodium.',
    'ravioli': '- Food: Ravioli\n- Calories: 220 per cup\n- Health Score: 5/10\n- Summary: Pasta with filling, moderate calories but refined carbs.',
    'red_velvet_cake': '- Food: Red Velvet Cake\n- Calories: 390 per slice\n- Health Score: 3/10\n- Summary: Colorful cake high in sugar and artificial ingredients.',
    'risotto': '- Food: Risotto\n- Calories: 320 per cup\n- Health Score: 5/10\n- Summary: Creamy rice dish high in calories but can include vegetables.',
    'samosa': '- Food: Samosa\n- Calories: 150 per samosa\n- Health Score: 5/10\n- Summary: Fried pastry with vegetables, moderate calories.',
    'sashimi': '- Food: Sashimi\n- Calories: 120 per serving\n- Health Score: 9/10\n- Summary: Pure fish protein, very healthy and low calorie.',
    'scallops': '- Food: Scallops\n- Calories: 140 per serving\n- Health Score: 8/10\n- Summary: Lean seafood protein low in calories and fat.',
    'seaweed_salad': '- Food: Seaweed Salad\n- Calories: 70 per serving\n- Health Score: 9/10\n- Summary: Very low calorie, rich in minerals and vitamins.',
    'shrimp_and_grits': '- Food: Shrimp and Grits\n- Calories: 420 per serving\n- Health Score: 6/10\n- Summary: Good protein from shrimp but high in calories.',
    'spaghetti_bolognese': '- Food: Spaghetti Bolognese\n- Calories: 380 per serving\n- Health Score: 6/10\n- Summary: Balanced meal with protein but high in refined carbs.',
    'spaghetti_carbonara': '- Food: Spaghetti Carbonara\n- Calories: 450 per serving\n- Health Score: 4/10\n- Summary: Rich pasta dish high in calories and saturated fat.',
    'spring_rolls': '- Food: Spring Rolls\n- Calories: 100 per roll\n- Health Score: 7/10\n- Summary: Light appetizer with vegetables, healthy when fresh.',
    'steak': '- Food: Steak\n- Calories: 350 per 6 oz\n- Health Score: 6/10\n- Summary: High protein and iron but can be high in saturated fat.',
    'strawberry_shortcake': '- Food: Strawberry Shortcake\n- Calories: 320 per serving\n- Health Score: 4/10\n- Summary: Sweet dessert high in sugar but contains some fruit.',
    'sushi': '- Food: Sushi\n- Calories: 40 per piece\n- Health Score: 8/10\n- Summary: Healthy combination of rice, fish, and vegetables.',
    'tacos': '- Food: Tacos\n- Calories: 150 per taco\n- Health Score: 6/10\n- Summary: Can be healthy with lean protein and vegetables.',
    'takoyaki': '- Food: Takoyaki\n- Calories: 40 per ball\n- Health Score: 5/10\n- Summary: Japanese octopus balls, moderate calories when not fried.',
    'tiramisu': '- Food: Tiramisu\n- Calories: 350 per serving\n- Health Score: 3/10\n- Summary: Italian dessert high in sugar, cream, and calories.',
    'tuna_tartare': '- Food: Tuna Tartare\n- Calories: 160 per serving\n- Health Score: 8/10\n- Summary: Fresh raw fish high in protein and omega-3s.',
    'waffles': '- Food: Waffles\n- Calories: 220 per waffle\n- Health Score: 4/10\n- Summary: Breakfast treat high in refined flour and sugar.'
}

def get_health_info(food_name: str) -> str:
    return nutrition_data.get(food_name, "⚠️ No nutrition data available for this item.")

# === Load Classes ===
with open(CLASS_TXT_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found at:", MODEL_PATH)
    exit()

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 101)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)
print("✅ Model loaded.")

# === Open Camera ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Could not open webcam. Trying alternatives...")
    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Opened camera at index {i}")
            break
    else:
        print("❌ No webcam found.")
        exit()

print("📷 Press 's' to snap | 'r' to reset | 'q' to quit")

# === Main Loop ===
DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 720
snapped = False
snap_frame = None
nutrition_info = ""
detected_label = ""
detected_conf = 0
health_cache = {}

while True:
    if not snapped:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break
        snap_display = frame.copy()
    else:
        snap_display = snap_frame.copy()

    display_frame = cv2.resize(snap_display, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    if not snapped:
        cv2.putText(display_frame, "📷 Press 's' to SNAP", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        h, w, _ = snap_frame.shape
        box_size = min(h, w)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cropped = snap_frame[y1:y2, x1:x2]
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)
            max_prob, pred = torch.max(prob, 1)

        detected_label = class_names[pred.item()]
        detected_conf = max_prob.item()

        if detected_label in health_cache:
            nutrition_info = health_cache[detected_label]
        else:
            nutrition_info = get_health_info(detected_label)
            health_cache[detected_label] = nutrition_info

        label_text = f"{detected_label.replace('_', ' ').title()} ({detected_conf:.2f})"
        cv2.putText(display_frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if nutrition_info:
            for i, line in enumerate(nutrition_info.splitlines()):
                cv2.putText(display_frame, line.strip(), (40, 100 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display_frame, "🔁 Press 'r' to re-snap", (20, DISPLAY_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("🍽️ Food Nutrition Snap", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not snapped:
        snap_frame = frame.copy()
        snapped = True
    elif key == ord('r'):
        snapped = False
        nutrition_info = ""
        detected_label = ""

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting...")