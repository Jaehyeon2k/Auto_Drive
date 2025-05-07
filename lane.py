import cv2
import numpy as np

# 영상 불러오기
cap = cv2.VideoCapture("road.mp4")  # road.mp4라는 동영상 파일을 읽음

# 관심 영역(ROI) 설정 함수 (도로 영역만 남기기 위해 마스크 생성)
def region_of_interest(image):
    height, width = image.shape[:2]  # 프레임의 세로(height)와 가로(width) 크기 가져오기
    # 도로 영역을 정의하는 다각형 좌표 설정 (좌우 넓고 상단을 낮게 조정하여 먼 거리 줄임)
    polygons = np.array([[
        (int(0.0 * width), height),              # 왼쪽 아래
        (int(1.0 * width), height),              # 오른쪽 아래
        (int(0.75 * width), int(height * 0.55)),  # 오른쪽 위 (낮춤)
        (int(0.25 * width), int(height * 0.55))   # 왼쪽 위 (낮춤)
    ]], np.int32)
    mask = np.zeros_like(image)  # 이미지와 동일한 크기의 0으로 채워진 마스크 생성
    cv2.fillPoly(mask, polygons, 255)  # 위에서 정의한 다각형 부분만 흰색(255)으로 채움
    return cv2.bitwise_and(image, mask)  # 마스크와 원본 이미지를 AND 연산하여 ROI만 남김

# 차선 영상 위에 선을 그리는 함수
def draw_lines(image, lines):
    line_image = np.zeros_like(image)  # 입력 이미지와 같은 크기의 빈 이미지 생성
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # 수직 또는 너무 짧거나 긴 선 제외
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if 30 < length < 300 and abs(y2 - y1) > 20 and abs(x2 - x1) > 20:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # 녹색 선으로 그림
    return line_image

# 영상 프레임 반복 처리
while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 프레임이 없으면 종료

    # 전처리 단계
    frame = cv2.resize(frame, (640, 360))  # 영상 해상도 줄이기
    blur = cv2.GaussianBlur(frame, (5, 5), 0)  # 가우시안 블러 적용 (노이즈 제거)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    edges = cv2.Canny(gray, 50, 150)  # Canny 엣지 검출 (하한값 50, 상한값 150으로 상향)

    # 관심 영역 마스크 적용
    roi = region_of_interest(edges)

    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLinesP(
        roi,                 # 입력 이미지 (ROI 영역)
        1,                  # 거리 해상도
        np.pi / 180,        # 각도 해상도 (1도)
        threshold=40,       # 직선으로 판단하기 위한 최소 교차점 수
        minLineLength=50,   # 최소 직선 길이
        maxLineGap=30       # 선 간 최대 허용 간격
    )

    # 선을 원본 영상 위에 그림
    line_image = draw_lines(frame, lines)
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # 원본과 선을 합성 (가중치 조절)

    # 결과 화면 출력
    cv2.imshow("Lane Detection", combo)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
