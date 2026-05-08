# Weakly Supervised YOLO Network for Surgical Instrument Localization in Endoscopic Videos

Rongfeng Wei, Jinlin Wu, Xuexue Bai, Ming Feng, Zhen Lei, Hongbin Liu, and Zhen Chen (2024)

## 🧩 Problem to Solve

본 논문은 최소 침습 수술(minimally invasive surgery)에서 내시경 비디오 내 수술 도구의 정확한 위치를 찾아내는 **Surgical Instrument Localization** 문제를 다룬다. 수술 도구의 위치를 정확히 파악하는 것은 수술의 품질과 안전성을 높이는 데 매우 중요하다.

그러나 내시경 비디오에서 도구의 위치를 위해 Bounding Box를 직접 생성하는 수동 어노테이션 작업은 매우 지루하고 노동 집약적이며 비용이 많이 든다. 반면, da Vinci 로봇 수술 시스템과 같은 장비는 센서를 통해 어떤 도구가 사용되었는지에 대한 카테고리 정보(category information)와 장착/해제 시점 등의 타임스탬프 정보를 자동으로 기록할 수 있어, 이러한 정보는 비교적 쉽게 얻을 수 있다.

따라서 본 논문의 목표는 정밀한 위치 정보 대신 **단순한 도구 카테고리 정보만을 약한 지도(weak supervision)로 활용하여 수술 도구의 위치를 추정하는 WS-YOLO 프레임워크를 제안하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **카테고리 독립적인 도구 부품 탐지(Category-free instrument localization)를 통해 초기 가이드라인을 잡고, 이를 기반으로 다회차 반복 학습(Multi-round training)을 통해 유사 라벨(Pseudo-labels)을 정제해 나가는 것**이다.

구체적인 설계 직관은 다음과 같다.

1. 어떤 도구든 공통적으로 가지는 부품(shaft, clevis, tip)을 먼저 탐지하여 초기 위치 후보군을 생성한다.
2. 비디오 캡션에 나타난 도구의 순서가 일반적으로 왼쪽에서 오른쪽 방향의 배치 순서와 일치한다는 경험적 관찰을 활용하여 초기 라벨을 할당한다.
3. 도구 전체를 탐지하는 모델과 부품을 탐지하는 모델 간의 위치 일관성(Location Consistency)을 검사하여 노이즈가 섞인 유사 라벨을 필터링한다.

## 📎 Related Works

논문에서는 직접적인 관련 연구 리스트를 길게 나열하지는 않았으나, 다음과 같은 배경 지식과 데이터셋을 언급하고 있다.

- **SIMS 데이터셋**: 수술 도구의 여러 부품(shaft, clevis, tip)에 대한 시맨틱 마스크 정보를 제공하는 데이터셋으로, 본 연구에서는 이를 활용해 카테고리 독립적인 부품 탐지 모델을 학습시킨다.
- **Endoscopic Vision Challenge 2023 데이터셋**: 제안 방법론을 검증하기 위해 사용된 벤치마크 데이터셋이다.
- **YOLOv8**: 실시간 객체 탐지 모델로, 본 프레임워크의 기본 아키텍처로 채택되었다.

기존 접근 방식과의 차별점은 수동으로 정밀하게 작성된 Bounding Box 어노테이션 없이, 로봇 시스템에서 얻을 수 있는 카테고리 정보와 공공 데이터셋의 부품 정보를 조합하여 위치 추정 성능을 단계적으로 끌어올렸다는 점이다.

## 🛠️ Methodology

WS-YOLO 프레임워크는 크게 **위치 초기화(Localization Initialization)** 단계와 **다회차 학습(Multi-round training)** 단계로 구성된다.

### 1. Localization Initialization

먼저, 특정 도구의 종류와 상관없이 도구의 공통 부품인 **shaft, clevis, tip** 세 가지를 탐지하는 모델 $\text{Det}_{\text{parts}}$를 학습시킨다.

- **학습 데이터**: SIMS 데이터셋의 시맨틱 마스크를 Bounding Box 형태로 변환하여 사용한다.
- **모델**: YOLOv8 기반의 아키텍처를 사용한다.
- **역할**: 이 모델은 이후 단계에서 도구의 종류는 모르지만 "여기에 도구가 있다"라는 초기 위치 정보를 제공하는 역할을 한다.

### 2. Multi-round Training

초기화된 $\text{Det}_{\text{parts}}$ 모델을 내시경 비디오 클립에 적용하여 Bounding Box를 생성한 후, 다음의 전략으로 도구별 위치 모델 $\text{Det}_{\text{tools}}$를 학습시킨다.

#### (1) 초기 라벨 할당

저자들은 도구들이 서로 겹치지 않았을 때, 캡션에 기록된 도구 순서와 화면상의 왼쪽$\rightarrow$오른쪽 배치 순서가 일치한다는 점에 주목하였다. 따라서 탐지된 박스가 3개인 프레임을 선택하고, 왼쪽에서 오른쪽 순서대로 캡션의 도구 이름을 할당하여 1차 유사 라벨 데이터셋을 구축한다.

#### (2) 유사 라벨 필터링 (Algorithm 1)

학습 과정에서 발생하는 노이즈를 제거하기 위해 $\text{Det}_{\text{tools}}$(도구 탐지기)와 $\text{Det}_{\text{parts}}$(부품 탐지기)의 결과 사이의 **IoU(Intersection over Union)**를 통해 일관성을 검사한다.

필터링 조건은 다음과 같다.
$$ \text{IoU}(\text{bbox}_{\text{part}}, \text{bbox}_{\text{tool}}) > \tau $$
여기서 임계값 $\tau$는 $0.8$로 설정되었다.

특히, 도구의 특성에 따라 서로 다른 부품과 매칭시킨다.

- **SpecialList**에 포함된 도구(예: monopolar curved scissors, stapler 등) $\rightarrow$ **tip** 부품의 위치와 비교.
- 그 외의 일반 도구 $\rightarrow$ **clevis** 부품의 위치와 비교.

만약 한 이미지 내의 모든 탐지된 도구 박스($\text{BBox}_{\text{tools}}$)가 이 일관성 검사를 통과하면, 해당 프레임은 고품질의 유사 라벨로 간주되어 다음 라운드의 학습 데이터로 사용된다.

#### (3) 반복 학습 과정

이 과정은 다음과 같이 순환한다.
$\text{Initial Labels} \rightarrow \text{Train } \text{Det}_{\text{tools}} \rightarrow \text{Filter using } \text{Det}_{\text{parts}} \rightarrow \text{Updated Labels} \rightarrow \text{Train } \text{Det}_{\text{tools}} \dots$

## 📊 Results

실험은 **Endoscopic Vision Challenge 2023** 데이터셋에서 수행되었으며, 지표로는 $\text{mAP}@[.5:.05:0.95]$를 사용하였다.

### 정량적 결과

다회차 학습이 반복될수록 mAP 수치가 지속적으로 상승하는 긍정적 피드백 루프(positive feedback mechanism)가 관찰되었다.

| Iteration | mAP (%) |
| :--- | :--- |
| 0 (Initial) | 4.3 |
| 1 | 10.9 |
| 2 | 13.5 |
| 3 | 14.7 |
| 4 | 15.7 |

초기 라운드(Iteration 0)에서는 4.3%에 불과했던 성능이 4회차 반복 학습 후에는 15.7%까지 향상되었다. 이는 제안한 유사 라벨 필터링 전략이 약한 지도 환경에서도 모델의 성능을 유의미하게 개선할 수 있음을 보여준다.

## 🧠 Insights & Discussion

**강점 및 해석**

- 본 연구는 사람이 직접 박스를 그리는 고비용 작업 없이, 기존의 부품 탐지 모델과 도구 카테고리 정보만을 결합하여 위치 탐지 모델을 구축했다는 점에서 실용성이 높다.
- 특히 $\text{Det}_{\text{parts}}$와 $\text{Det}_{\text{tools}}$라는 두 모델의 상호 보완적인 관계를 이용하여 유사 라벨의 순도를 높이는 전략이 유효했음을 알 수 있다.

**한계 및 가정**

- **순서 가정**: 초기 라벨을 할당할 때 '왼쪽에서 오른쪽'이라는 단순한 기하학적 순서에 의존한다. 도구가 심하게 겹쳐 있거나 순서가 뒤바뀐 경우 초기 라벨의 오염이 심할 수 있으며, 이는 초기 성능(4.3%)이 매우 낮은 원인으로 분석된다.
- **성능 수치**: 반복 학습을 통해 성능이 향상되었음에도 불구하고, 최종 mAP 15.7%는 완전 지도 학습(fully supervised) 결과와 비교했을 때 여전히 낮을 가능성이 크다. 약한 지도 학습의 한계로 보이며, 더 정교한 유사 라벨 생성 기법이 필요할 수 있다.
- **데이터셋 의존성**: SIMS 데이터셋을 통해 사전 학습된 $\text{Det}_{\text{parts}}$ 모델의 성능이 WS-YOLO 전체 성능의 하한선을 결정짓는 중요한 요소가 된다.

## 📌 TL;DR

본 논문은 수술 도구 위치 탐지를 위해 정밀한 Bounding Box 어노테이션 대신, 쉽게 얻을 수 있는 도구 카테고리 정보와 부품 탐지 모델을 활용하는 **WS-YOLO** 프레임워크를 제안한다. 부품-도구 간의 위치 일관성을 이용해 유사 라벨을 정제하는 다회차 학습 전략을 통해, 추가적인 수동 라벨링 없이도 위치 탐지 성능을 점진적으로 향상시켰다. 이 연구는 의료 영상 분야에서 라벨링 비용 문제를 해결하고 로봇 수술 시스템의 자동화된 분석 도구를 개발하는 데 기여할 수 있다.
