# D’RespNeT: A UAV Dataset and YOLOv8-DRN Model for Aerial Instance Segmentation of Building Access Points for Post-Earthquake Search-and-Rescue Missions

Aykut Sirma, Angelos Plastropoulos, Gilbert Tang, Argyrios Zolotas (2025)

## 🧩 Problem to Solve

본 연구는 지진 발생 후 초기 대응 단계인 '골든 타임(Golden 48-hour window)' 내에 인명 구조 효율을 극대화하기 위해, 무인 항공기(UAV)를 이용한 신속하고 정밀한 환경 분석 문제를 해결하고자 한다. 특히 지진으로 파괴된 도시 환경에서 구조 대원이나 무인 지상 차량(UGV)이 진입할 수 있는 '접근 가능 지점(Access Points)'을 실시간으로 식별하는 것이 핵심이다.

기존의 재난 관련 데이터셋들은 다음과 같은 한계가 존재한다. 첫째, 위성 이미지 기반 데이터셋(예: xBD)은 광범위한 지역을 커버하지만 해상도가 낮아 건물 외벽 수준의 진입로 식별이 불가능하다. 둘째, 기존의 UAV 데이터셋(예: FloodNet)은 특정 재난(홍수 등)에 치중되어 있거나, 단순히 Semantic Segmentation(의미론적 분할) 또는 Bounding-box 기반의 Object Detection만을 제공한다. Semantic Segmentation은 인접한 문이나 창문을 하나의 클래스로 묶어버리기 때문에, 개별 진입로의 상태를 개별적으로 평가해야 하는 구조 작전(Triage)에 부적합하며, Bounding-box는 로봇의 정밀한 경로 계획이나 조작에 필요한 기하학적 디테일을 제공하지 못한다. 따라서 본 논문의 목표는 고해상도 UAV 영상을 활용하여 개별 객체를 픽셀 단위로 구분하는 Instance Segmentation 데이터셋을 구축하고, 이를 통해 실시간으로 작동하는 고성능 탐지 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 재난 대응을 위한 전문적인 Instance Segmentation 데이터셋인 D’RespNeT를 제안하고, 이를 최적화한 YOLOv8-DRN 모델을 구현한 것이다. 중심적인 설계 아이디어는 다음과 같다.

1. **D’RespNeT 데이터셋 구축**: 실제 지진 피해 지역(튀르키예, 미얀마 등)의 1080p 고해상도 UAV 영상을 기반으로, 구조 작전에 필수적인 28개 클래스에 대해 Polygon-level의 정밀한 Instance Segmentation 마스크를 제공한다. 특히 '접근 가능(Accessible)'과 '차단됨(Blocked)'을 명확히 구분하여 작전 효율성을 높였다.
2. **실시간 Instance Segmentation 모델**: YOLOv8-seg를 기반으로 최적화된 YOLOv8-DRN 모델을 통해 RTX-4090 GPU 기준 27 FPS의 추론 속도와 $92.7\%$의 $\text{mAP}_{50}$를 달성하여, 실시간 상황 인식 요구사항을 충족하였다.
3. **지각-제어 통합 파이프라인**: 단순히 객체를 검출하는 것에 그치지 않고, 예측된 마스크를 로봇이 이용 가능한 Cost-map으로 변환하고, 이를 Reinforcement Learning(강화학습)과 연계하여 실제 UGV의 경로 계획 성능을 향상시키는 프레임워크를 제시하였다.

## 📎 Related Works

논문은 기존의 재난 대응 데이터셋을 위성 기반과 UAV 기반으로 나누어 설명하며 각각의 한계를 지적한다.

- **위성 기반 접근 방식 (xBD, HRUD 등)**: 넓은 지역의 피해 규모를 파악하는 데는 유리하나, 공간 해상도가 낮아 개별 건물의 진입구(Entry point)를 식별하는 정밀한 로봇 구조 작업에는 적용하기 어렵다.
- **UAV 기반 접근 방식 (FloodNet, AIDER 등)**: 해상도는 높지만, 주로 홍수 피해에 집중되어 있거나 단순한 Semantic Segmentation 레이블만을 제공한다. 특히 Semantic Segmentation은 동일 클래스의 인접 객체들을 하나로 병합하기 때문에, 개별 진입로의 상태를 판단해야 하는 SAR(Search-and-Rescue) 작전의 '객체 중심적(Object-centric)' 특성을 반영하지 못한다.

D’RespNeT는 이러한 한계를 극복하기 위해 **Instance Segmentation** 방식을 채택하였다. 이를 통해 각 진입로의 개별 identity를 유지함으로써, 로봇이 각 입구의 기하학적 구조를 파악하고 정밀한 접근 벡터를 계산할 수 있도록 하여 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 1. 데이터 수집 및 구축

- **데이터 소스**: 튀르키예 지진 및 기타 분쟁/재난 지역의 1080p UAV 영상을 수집하였다.
- **전처리**: 원본 $1920 \times 1080$ 이미지를 YOLOv8 입력 규격인 $640 \times 640$으로 리사이징하였다.
- **데이터 증강**: 164장의 원본 이미지에 Horizontal flip, $\pm 15^\circ$ 회전, $\pm 10^\circ$ Shear, 색상 지터링(Colour jitter), Gaussian blur 등을 적용하여 총 820장의 학습 데이터를 생성하였다.
- **어노테이션**: Roboflow 플랫폼을 사용하여 28개 클래스에 대해 Polygon-level 마스크와 Bounding-box를 생성하였다.

### 2. 클래스 체계

데이터셋은 크게 5가지 카테고리로 구분된다.

- **건물 구조 상태**: 붕괴(Collapsed), 손상(Damaged), 미손상(Undamaged).
- **진입 가능성**: 문, 창문, 틈새의 '접근 가능' 및 '차단됨' 상태 구분.
- **잔해 및 장애물**: 잔해의 정도(Heavy, Moderate, Light) 및 파편(Rubble).
- **인적 요소**: 민간인, 구조대원.
- **차량 및 인프라**: 굴착기, 트럭 등 중장비와 도로, 교량, 안전한 드론 착륙 지점.

### 3. 모델 아키텍처 및 학습 절차

- **기본 모델**: 실시간성과 정확도의 균형을 위해 $\text{YOLOv8-seg}$ (Large variant)를 채택하였다.
- **학습 방식**: COCO 사전 학습 가중치에서 시작하여 fine-tuning을 진행하였으며, 총 7차례의 연속적인 학습(weight continuation)을 통해 성능을 점진적으로 향상시켰다.
- **강화학습(RL) 미세 조정**: 지각 모델의 출력을 제어 단계와 연결하기 위해 PPO(Proximal Policy Optimization) 알고리즘을 사용한 RL 단계를 추가하였다. 이때의 보상 함수(Reward function)는 다음과 같다.
$$R = \alpha N_{\text{victims}} - \beta t_{\text{coll}} - \gamma E_{\text{GPU}}$$
여기서 $N_{\text{victims}}$는 도달한 생존자 수, $t_{\text{coll}}$은 충돌 시간, $E_{\text{GPU}}$는 GPU 에너지 소비량이며, $\alpha, \beta, \gamma$는 각각의 가중치이다.

### 4. 지각-비용 지도(Perception $\rightarrow$ Cost-Map) 파이프라인

검출된 폴리곤 마스크는 다음 과정을 거쳐 UGV의 주행 지도로 변환된다.

1. **Rasterization**: 마스크를 $1\text{cm}/\text{px}$ 점유 그리드(Occupancy grid) $O$로 변환한다.
2. **Height Estimation**: 잔해 마스크의 면적-그림자 휴리스틱을 통해 대략적인 높이 $h(x, y)$를 추정하여 2.5D 레이어 $H$를 생성한다.
3. **Fusion**: $O, H$ 및 LiDAR/SLAM 데이터를 베이지안 증거 누적(Bayesian evidence accumulation) 방식으로 융합하여 비용 레이어 $C(x, y) \in \{0, 1, 2, 3\}$를 생성한다. ($0$: 자유 구역, $1$: 주의, $2$: 제거 필요, $3$: 통과 불가)

## 📊 Results

### 1. 정량적 성능 평가

- **정확도**: 최종 모델은 $\text{mAP}_{50} = 92.7\%$, $\text{Precision} = 83.2\%$, $\text{Recall} = 87.7\%$를 달성하였다.
- **추론 속도**: RTX-4090 GPU에서 TensorRT FP16 가속 적용 시 프레임당 37ms, 즉 **27 FPS**의 속도로 추론이 가능하여 실시간 운영 기준(50ms 미만)을 충족하였다.
- **정밀도**: 진입로 클래스에 대한 Boundary-IoU가 $0.91$로 측정되어, $\pm 5\text{px}$의 공학적 허용 오차 범위 내에 들어왔음을 확인하였다.

### 2. 작전적 영향 평가

- **RL 적용 결과**: RL 미세 조정을 통해 특정 소수 클래스(차단된 창문 등)의 $\text{mAP}_{50}$가 $1.8\text{pp}$ 향상되었으며, 평균 미션 수행 시간이 $17\%$ 단축되었다.
- **인간-로봇 인터페이스(HRI) 평가**: 11명의 인증된 구조 전문가를 대상으로 한 NASA-TLX 실험 결과, Instance-level 오버레이를 제공했을 때 인지적 작업 부하가 $19\%$ 감소하였으며, 진입로 지정에 필요한 상호작용 횟수가 절반으로 줄어들었다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

- **Instance-level의 중요성**: 본 연구는 단순한 Semantic Segmentation보다 Instance Segmentation이 왜 필수적인지를 입증하였다. 개별 객체의 ID를 유지함으로써 인접한 진입로 간의 거리 계산, 개별 객체의 기하학적 쿼리(표면 법선, 간격)가 가능해지며, 이는 곧 로봇의 실제 주행 경로 계획(Path planning) 성능 향상으로 직결된다.
- **데이터 질의 승리**: 205장의 원본 이미지라는 상대적으로 적은 양의 데이터임에도 불구하고, 정밀한 Dual-pass 검수와 전략적 증강을 통해 대규모 데이터셋에 필적하는 높은 성능을 낼 수 있음을 보여주었다.

### 2. 한계점 및 비판적 해석

- **클래스 불균형 (Long-tail distribution)**: 굴착기(Excavator)나 차단된 문(Blocked door)과 같은 소수 클래스의 데이터가 부족하여, Confusion Matrix 상에서 일부 오검출(특히 차단된 문 $\rightarrow$ 차단된 창문)이 관찰된다.
- **환경적 편향**: 모든 데이터가 주간 및 맑은 날씨에 촬영되었으므로, 야간이나 안개, 연기가 자욱한 실제 재난 현장에서의 일반화 성능은 아직 검증되지 않았다.
- **센서 단일성**: RGB 영상에만 의존하고 있어, 열화상 카메라나 LiDAR와의 다중 모달(Multimodal) 융합을 통한 생존자 탐색 기능까지는 확장되지 않았다.

## 📌 TL;DR

본 논문은 지진 후 구조 작전을 위해 UAV 기반의 고해상도 Instance Segmentation 데이터셋인 **D’RespNeT**와 실시간 탐지 모델인 **YOLOv8-DRN**을 제안한다. 28개 클래스에 대한 정밀한 폴리곤 마스크를 통해 진입로의 접근 가능 여부를 판단하며, 이를 비용 지도(Cost-map)로 변환해 UGV의 경로 계획에 활용하는 파이프라인을 구축하였다. $\text{mAP}_{50} = 92.7\%$와 $27\text{FPS}$라는 탁월한 성능을 통해 실시간 재난 대응 가능성을 입증했으며, 이는 향후 무인 시스템의 자율 구조 작전 및 인간-로봇 협업 능력을 향상시키는 데 중요한 기반이 될 것으로 보인다.
