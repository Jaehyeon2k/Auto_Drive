import cv2
import numpy as np

# 영상 불러오기
cap = cv2.VideoCapture("road.mp4")

# 관심 영역(ROI) 설정 함수 (왼쪽 도로까지 포함하도록 확장)
def region_of_interest(image):
    height, width = image.shape[:2]
    polygons = np.array([[
        (int(0.0 * width), height),
        (int(1.0 * width), height),
        (int(0.6 * width), int(height * 0.45)),
        (int(0.3 * width), int(height * 0.45))
    ]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

# 차선 영상 위에 선을 그리는 함수
def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if abs(y2 - y1) > 20:  # 수직 또는 너무 짧은 선 제외
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

# 프레임 반복 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 전처리: 크기 조정, 흑백 변환, Canny 엣지 검출
    frame = cv2.resize(frame, (640, 360))
    blur = cv2.GaussianBlur(frame, (5, 5), 0)  # 노이즈 제거를 위한 블러 추가
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)

    # 관심 영역(도로 부분) 추출
    roi = region_of_interest(edges)

    # 직선 검출 (Hough Transform)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=40, minLineLength=50, maxLineGap=30)

    # 선 그리기
    line_image = draw_lines(frame, lines)
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # 결과 출력
    cv2.imshow("Lane Detection", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()