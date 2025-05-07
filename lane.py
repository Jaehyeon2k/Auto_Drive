import cv2               # OpenCV: 이미지/영상 처리 라이브러리
import numpy as np       # NumPy: 수치 계산과 배열 연산용 라이브러리

# ▶ 도로 영상 파일 불러오기
cap = cv2.VideoCapture("road.mp4")  # 같은 폴더 내 road.mp4 영상 사용

# ▶ 관심 영역(ROI)을 설정하는 함수 정의
def region_of_interest(image):
    height = image.shape[0]              # 영상 높이 추출
    width = image.shape[1]               # 영상 너비 추출

    # 삼각형 영역 지정 (도로 아래쪽만 남기기)
    polygons = np.array([[
        (100, height),                   # 좌측 하단
        (width - 100, height),          # 우측 하단
        (width // 2, int(height * 0.6)) # 중앙 상단 꼭짓점
    ]], np.int32)

    mask = np.zeros_like(image)         # 영상과 동일한 크기의 검정색 마스크 생성
    cv2.fillPoly(mask, polygons, 255)   # 다각형 부분만 흰색으로 채움

    # 원본 이미지와 마스크를 AND 연산 → ROI 영역만 남김
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# ▶ 영상이 열려 있는 동안 반복 실행
while cap.isOpened():
    ret, frame = cap.read()             # 한 프레임 읽기
    if not ret:
        break                           # 프레임이 없으면 종료

    # 프레임 크기 조절 (가로 640, 세로 360)
    frame = cv2.resize(frame, (640, 360))

    # 컬러 영상 → 흑백 영상 변환 (엣지 검출을 위해)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 윤곽선(Edge) 검출: 흰 선은 경계, 검정은 배경
    edges = cv2.Canny(gray, 30, 100)

    # 관심 영역(도로 부분만) 추출
    roi = region_of_interest(edges)

    # 결과 영상 출력 (새 창으로 띄움)
    cv2.imshow("Lane Detection", roi)

    # 'q' 키를 누르면 영상 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ▶ 영상 종료 후 자원 정리
cap.release()
cv2.destroyAllWindows()
