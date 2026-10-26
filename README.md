# 지하철 실내 길찾기 (Subway Indoor Navi)

외국인 방문객이 지하철역 안에서 길을 잃지 않도록 돕는 **무료·API 없는** 실내 내비게이션입니다.

1. 관리자가 역 안을 휴대폰 동영상으로 걸으며 촬영합니다.
2. AI가 영상에서 지점을 뽑고, **겹치는 장면을 자동으로 찾아** 역 전체 길 지도를 만듭니다.
3. 사용자가 주변(표지판, 출구 번호 등)을 사진으로 찍으면 현재 위치를 찾습니다.
4. 목적지를 고르면 최단 경로를 **사진 단계별로** 안내합니다 (한국어/English/中文/日本語).

모든 AI 계산은 PC와 사용자 휴대폰 브라우저 안에서 실행됩니다. 서버, 유료 API, 앱 설치가 필요 없습니다.

```
mapper/            PC에서 실행하는 지도 만들기 (Python)
  export_model.py    무료 AI 모델(MobileNetV3) → web/model/embed.onnx (최초 1회)
  build_map.py       동영상 → web/data/<역>/graph.json + 사진
stations/seoul/    서울역 원본 자료
  videos/            촬영한 동영상을 여기에 넣기 (git에는 안 올라감)
  labels.csv         목적지 이름표
web/               휴대폰용 길찾기 웹페이지 (GitHub Pages로 배포)
```

## 1. 설치 (최초 1회)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r mapper/requirements.txt
.\.venv\Scripts\python mapper/export_model.py
```

## 2. 서울역 촬영 방법

- 경로마다 영상 1개씩 찍습니다. 예: `route1.mp4` = 1호선 승강장 → 1번 출구
- **천천히** 걷고, 휴대폰을 가슴 높이에서 **정면**으로 듭니다.
- 갈림길, 개찰구, 계단 앞에서는 2~3초 멈추고 주변을 천천히 한 바퀴 찍습니다.
- **영상끼리 겹치는 구간이 있어야** 자동으로 이어집니다. 예: route2를 route1의 중간 지점(환승통로)에서 시작하기.
- 가능하면 같은 길을 반대 방향으로도 찍습니다. 돌아보는 방향의 풍경이 다르기 때문입니다.
- 표지판, 출구 번호판, 노선 색 안내를 잘 보이게 찍으면 위치 정확도가 크게 올라갑니다.

## 3. 목적지 이름 붙이기

영상을 보며 목적지가 나오는 시각(초)을 `stations/seoul/labels.csv`에 적습니다.

```csv
video,time,ko,en,zh,ja
route1.mp4,0,1호선 승강장,Line 1 Platform,1号线站台,1号線ホーム
route1.mp4,95,1번 출구,Exit 1,1号出口,1番出口
```

## 4. 지도 만들기

```powershell
.\.venv\Scripts\python mapper/build_map.py --station seoul
```

"지도가 N조각으로 나뉘어 있습니다"라는 메시지가 나오면, 조각 사이를 겹치게 한 번 더 촬영해서 `videos/`에 추가하고 다시 실행합니다.

## 5. 휴대폰에서 시험하기

PC에서 확인:

```powershell
.\.venv\Scripts\python -m http.server 8000 -d web
# 브라우저에서 http://localhost:8000
```

휴대폰 카메라는 https 주소에서만 잘 작동하므로 GitHub Pages로 배포하는 것이 가장 간단합니다.

1. `web/data/seoul/`의 결과를 커밋하고 push합니다.
2. GitHub 저장소 **Settings → Pages → Source**를 **GitHub Actions**로 설정합니다 (최초 1회).
3. `https://bandit1397.github.io/indoor-tracking/` 에 접속합니다.

## 한계와 다음 단계

- 위치는 cm 단위가 아니라 "개찰구 앞" 같은 **지점 단위**입니다.
- 흰 벽, 타일만 보이는 복도는 서로 비슷해서 헷갈릴 수 있습니다. 표지판을 찍도록 안내합니다.
- 화면 속 사진에 사람 얼굴은 자동으로 흐리게 처리됩니다. 공개 서비스로 운영하려면 역 운영사(코레일, 서울교통공사, 공항철도)에 촬영 및 이용 허가를 확인하세요.
- 다음 단계 아이디어: COLMAP으로 3D 지도 복원, 화살표 AR 안내, 여러 역 지원(`?station=이름`).
