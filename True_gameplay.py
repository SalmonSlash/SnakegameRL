import numpy as np
import cv2
import mss
import mss.tools
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyautogui
import concurrent.futures

GRID_COLS = 17
GRID_ROWS = 15


def capture_screen(bbox=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if bbox is None else {"top": bbox[1], "left": bbox[0], "width": bbox[2]-bbox[0], "height": bbox[3]-bbox[1]}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def detect_game_board_region():

    screen_bgr = capture_screen()
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)

    lower_board = np.array([35, 140, 200])
    upper_board = np.array([45, 255, 255])
    mask_board = cv2.inRange(hsv, lower_board, upper_board)

    kernel = np.ones((3,3), np.uint8)
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_board = cv2.morphologyEx(mask_board, cv2.MORPH_OPEN, kernel, iterations=2)

    points = cv2.findNonZero(mask_board)
    if points is None:
        print("ไม่พบสีของกระดานในหน้าจอ")
        return None

    x, y, w, h = cv2.boundingRect(points)
    if w < 100 or h < 100:
        print("บริเวณที่ตรวจจับได้มีขนาดเล็กเกินไป")
        return None

    cv2.rectangle(screen_bgr, (x, y), (x + w, y + h), (0, 0, 255), 3)
    screen_rgb = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(screen_rgb)
    plt.title("Detected Game Board Region")
    plt.axis("off")
    plt.show()

    return (x, y, x + w, y + h)


def preprocess_image(image, resize_factor=0.5):

    new_dim = (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))
    image_resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
    return image_resized

def process_image_to_grid(image, grid_cols=GRID_COLS, grid_rows=GRID_ROWS):
    height, width, _ = image.shape
    cell_width = width / grid_cols
    cell_height = height / grid_rows
    grid = np.zeros((grid_rows, grid_cols), dtype=int)

    for row in range(grid_rows):
        for col in range(grid_cols):
            x_start = int(col * cell_width)
            x_end = int((col + 1) * cell_width)
            y_start = int(row * cell_height)
            y_end = int((row + 1) * cell_height)
            cell = image[y_start:y_end, x_start:x_end]
            cell_type = classify_cell_color(cell)
            grid[row, col] = cell_type
    return grid

def classify_cell_color(cell):
    cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

    lower_bg = np.array([35, 50, 100])
    upper_bg = np.array([45, 255, 255])
    mask_bg = cv2.inRange(cell_hsv, lower_bg, upper_bg)

    lower_snake = np.array([100, 50, 50])
    upper_snake = np.array([130, 255, 255])
    mask_snake = cv2.inRange(cell_hsv, lower_snake, upper_snake)

    lower_apple1 = np.array([0, 70, 50])
    upper_apple1 = np.array([10, 255, 255])
    lower_apple2 = np.array([170, 70, 50])
    upper_apple2 = np.array([180, 255, 255])
    mask_apple1 = cv2.inRange(cell_hsv, lower_apple1, upper_apple1)
    mask_apple2 = cv2.inRange(cell_hsv, lower_apple2, upper_apple2)
    mask_apple = cv2.bitwise_or(mask_apple1, mask_apple2)

    total_pixels = cell.shape[0] * cell.shape[1]
    ratio_bg = np.count_nonzero(mask_bg) / total_pixels
    ratio_snake = np.count_nonzero(mask_snake) / total_pixels
    ratio_apple = np.count_nonzero(mask_apple) / total_pixels

    threshold_ratio = 0.3
    if ratio_apple > threshold_ratio and ratio_apple >= ratio_snake and ratio_apple >= ratio_bg:
        return 2
    elif ratio_snake > threshold_ratio and ratio_snake >= ratio_apple and ratio_snake >= ratio_bg:
        return 1
    else:
        return 0

def overlay_grid_on_image(image, grid_cols=GRID_COLS, grid_rows=GRID_ROWS):
    height, width, _ = image.shape
    cell_width = width / grid_cols
    cell_height = height / grid_rows

    for col in range(1, grid_cols):
        x = int(col * cell_width)
        cv2.line(image, (x, 0), (x, height), (0, 255, 255), 1)
    for row in range(1, grid_rows):
        y = int(row * cell_height)
        cv2.line(image, (0, y), (width, y), (0, 255, 255), 1)
    return image

def find_snake_head_and_food(grid):
    snake_positions = list(zip(*np.where(grid == 1)))
    food_positions = list(zip(*np.where(grid == 2)))

    head_candidates = []
    for pos in snake_positions:
        r, c = pos
        neighbors = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == 1:
                    neighbors += 1
        if neighbors == 1:
            head_candidates.append(pos)
    if not head_candidates and snake_positions:
        head_candidates = snake_positions

    food = food_positions[0] if food_positions else None
    if food and head_candidates:
        head = min(head_candidates, key=lambda pos: abs(pos[0]-food[0])+abs(pos[1]-food[1]))
    elif head_candidates:
        head = head_candidates[0]
    else:
        head = None
    return head, food

def get_state(grid, head, food, current_direction):

    def cell_at(r, c):
        if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
            return 1
        return grid[r, c]

    if current_direction == 0:
        delta_straight = (-1, 0)
    elif current_direction == 1:
        delta_straight = (0, 1)
    elif current_direction == 2:
        delta_straight = (1, 0)
    elif current_direction == 3:
        delta_straight = (0, -1)

    right_direction = (current_direction + 1) % 4
    if right_direction == 0:
        delta_right = (-1, 0)
    elif right_direction == 1:
        delta_right = (0, 1)
    elif right_direction == 2:
        delta_right = (1, 0)
    elif right_direction == 3:
        delta_right = (0, -1)

    left_direction = (current_direction - 1) % 4
    if left_direction == 0:
        delta_left = (-1, 0)
    elif left_direction == 1:
        delta_left = (0, 1)
    elif left_direction == 2:
        delta_left = (1, 0)
    elif left_direction == 3:
        delta_left = (0, -1)

    head_r, head_c = head
    danger_straight = cell_at(head_r + delta_straight[0], head_c + delta_straight[1]) == 1
    danger_right = cell_at(head_r + delta_right[0], head_c + delta_right[1]) == 1
    danger_left = cell_at(head_r + delta_left[0], head_c + delta_left[1]) == 1

    dir_up = 1 if current_direction == 0 else 0
    dir_right = 1 if current_direction == 1 else 0
    dir_down = 1 if current_direction == 2 else 0
    dir_left = 1 if current_direction == 3 else 0

    food_left = 1 if food and food[1] < head_c else 0
    food_right = 1 if food and food[1] > head_c else 0
    food_up = 1 if food and food[0] < head_r else 0
    food_down = 1 if food and food[0] > head_r else 0

    state = np.array([
        int(danger_straight), int(danger_right), int(danger_left),
        dir_left, dir_right, dir_up, dir_down,
        food_left, food_right, food_up, food_down
    ], dtype=int)
    return state

def update_direction(current_direction, action):
    new_direction = (current_direction + (1 if action == 1 else -1 if action == 2 else 0)) % 4
    return new_direction

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def infer_action(model, state_tensor):
    with torch.no_grad():
        prediction = model(state_tensor)
    return torch.argmax(prediction).item()

def real_game_loop(model):
    print("เริ่มเล่นเกมจริงโดยใช้โมเดล...")
    board_bbox = detect_game_board_region()
    if board_bbox is None:
        print("ไม่พบตำแหน่งกระดานเกม")
        return
    print("ตำแหน่งกระดานเกม:", board_bbox)


    board_bgr = capture_screen(bbox=board_bbox)
    board_small = preprocess_image(board_bgr, resize_factor=0.5)
    grid = process_image_to_grid(board_small, GRID_COLS, GRID_ROWS)
    print("Grid ที่ตีความได้จากกระดานเกม:")
    print(grid)
    overlay_img = overlay_grid_on_image(board_small.copy(), GRID_COLS, GRID_ROWS)
    cv2.imshow("Debug: กระดานเกมพร้อม grid", overlay_img)
    print("กดปุ่มใดๆ บนหน้าต่างภาพเพื่อดำเนินการต่อ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    current_direction = 1
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    while True:
        try:
            start_time = time.time()
            board_bgr = capture_screen(bbox=board_bbox)

            board_small = preprocess_image(board_bgr, resize_factor=0.5)
            grid = process_image_to_grid(board_small, GRID_COLS, GRID_ROWS)

            head, food = find_snake_head_and_food(grid)
            if head is None:
                cv2.putText(board_bgr, "ไม่พบงู", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("Real Game with Grid", board_bgr)
                cv2.waitKey(2000)
                print("ไม่พบงูในเกม")
                break

            state = get_state(grid, head, food, current_direction)
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)


            future = executor.submit(infer_action, model, state_tensor)
            action = future.result()

            new_direction = update_direction(current_direction, action)

            if new_direction == 0:
                key = 'w'
            elif new_direction == 1:
                key = 'd'
            elif new_direction == 2:
                key = 's'
            elif new_direction == 3:
                key = 'a'
            else:
                key = None

            if key:
                pyautogui.press(key)
                print(f"กดปุ่ม: {key}, current_direction: {current_direction}, new_direction: {new_direction}, action: {action}")

            current_direction = new_direction

            overlay_image = overlay_grid_on_image(board_bgr.copy(), GRID_COLS, GRID_ROWS)
            cv2.putText(overlay_image, "Press 'q' to quit", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Real Game with Grid", overlay_image)

            elapsed = time.time() - start_time

            if cv2.waitKey(max(1, int(100 - elapsed*1000))) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Error:", e)
            break

    executor.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_size = 11
    hidden_size = 256
    output_size = 3
    model = LinearQNet(input_size, hidden_size, output_size)
    try:
        model.load_state_dict(torch.load("model_path.pth"))
        model.eval()
        print("โหลดโมเดลเรียบร้อยแล้ว")
    except Exception as e:
        print("ไม่สามารถโหลดโมเดลได้:", e)
        exit()
    real_game_loop(model)
