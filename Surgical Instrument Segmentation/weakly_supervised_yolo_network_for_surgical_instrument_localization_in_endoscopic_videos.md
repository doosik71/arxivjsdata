# Weakly Supervised YOLO Network for Surgical Instrument Localization in Endoscopic Videos

Rongfeng Wei, Jinlin Wu, Xuexue Bai, Ming Feng, Zhen Lei, Hongbin Liu, and Zhen Chen (2024)

## 🧩 Problem to Solve

최소 침습 수술(Minimally Invasive Surgery)에서 수술 도구를 정확하게 위치시키는 Localization 작업은 수술의 질과 안전성을 높이는 데 매우 중요하다. 그러나 내시경 영상 내에서 수술 도구의 Bounding Box를 수동으로 어노테이션(Annotation)하는 작업은 매우 지루하고 노동 집약적이며 높은 비용이 발생한다.

반면, da Vinci와 같은 로봇 수술 시스템은 센서를 통해 도구의 카테고리 정보와 장착/탈거 이벤트의 타임스탬프를 자동으로 기록할 수 있어, 카테고리 수준의 정보는 상대적으로 쉽게 획득 가능하다. 본 논문의 목표는 이러한 약한 감독(Weak Supervision) 정보인 도구 카테고리 정보를 활용하여, 정밀한 수동 어노테이션 없이도 수술 도구를 효과적으로 Localization 할 수 있는 WS-YOLO 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **카테고리 독립적인 부품 탐지(Category-free part localization)**와 **다단계 반복 학습 전략(Multi-round training strategy)**을 결합하여 점진적으로 정교한 Pseudo-label을 생성하고 이를 통해 모델을 학습시키는 것이다.

구체적으로, 도구의 종류와 상관없이 공통적으로 존재하는 부품(Shaft, Clevis, Tip)을 먼저 찾아내고, 이를 통해 생성된 초기 Pseudo-label을 반복적인 필터링 과정을 거쳐 정제함으로써 Localization 성능을 단계적으로 향상시키는 구조를 제안한다.

## 📎 Related Works

논문에서는 직접적인 관련 연구의 나열보다는 활용된 데이터셋과 베이스라인 모델을 중심으로 언급하고 있다.

- **SIMS 데이터셋**: 다양한 수술 도구의 세그멘테이션 마스크를 제공하는 공개 데이터셋으로, 본 연구에서는 이를 활용해 도구의 종류와 무관하게 부품(part)을 찾는 초기 모델을 학습시킨다.
- **YOLOv8**: 최신 객체 탐지 모델인 YOLOv8을 기반으로 Localization 모델을 구축하여 실시간성과 정확성을 확보하고자 하였다.
- **기존 방식과의 차별점**: 기존의 Localization 방식이 정밀하게 라벨링된 Bounding Box 데이터를 필요로 했다면, WS-YOLO는 센서 등을 통해 얻을 수 있는 단순 카테고리 정보만을 활용하여 학습 데이터를 스스로 생성하고 정제한다는 점에서 차별화된다.

## 🛠️ Methodology

WS-YOLO 프레임워크는 크게 'Localization 초기화' 단계와 '다단계 반복 학습' 단계로 구성된다.

### 1. Localization Initialization

먼저, SIMS 데이터셋의 세그멘테이션 마스크를 Bounding Box 형태로 변환하여, 도구의 카테고리 정보 없이 오직 부품(shaft, clevis, tip)만을 탐지하는 모델인 $\text{Det}_{\text{parts}}$를 학습시킨다. 이후 이 모델을 대상 내시경 영상에 적용하여 초기 Pseudo-label(Bounding Box들)을 생성한다.

### 2. Multi-round Training

초기 생성된 Pseudo-label은 노이즈가 많으므로, 이를 정제하기 위해 다음과 같은 전략을 사용한다.

**가. 초기 라벨 할당 (Initial Label Assignment)**
영상 캡션에 등장하는 도구의 순서가 일반적으로 영상 내 좌측에서 우측으로 배치되는 경향이 있다는 점과, 세 개 이상의 도구가 겹치는 경우는 드물다는 관찰 결과를 이용한다. 이에 따라 탐지된 박스가 3개인 프레임에 대해 좌측부터 우측 순으로 도구 이름을 할당하여 1차 $\text{Det}_{\text{tools}}$ 모델을 학습시킨다.

**나. 다단계 라벨 필터링 (Multi-round Label Filtering)**
학습된 $\text{Det}_{\text{tools}}$ 모델과 초기 $\text{Det}_{\text{parts}}$ 모델 간의 위치 일관성을 측정하여 노이즈를 제거한다. 필터링 기준은 다음과 같은 $\text{IoU}$(Intersection over Union) 수식을 따른다.

$$IOU(bbox_{part}, bbox_{tool}) > \tau$$

여기서 $\tau$는 $0.8$로 설정되었다. 구체적인 필터링 알고리즘(Algorithm 1)은 다음과 같다:

- 도구가 $\text{SpecialList}$(예: monopolar curved scissors 등)에 포함된 경우: $\text{Det}_{\text{parts}}$가 찾은 **tip** 부위와의 $\text{IoU}$를 확인한다.
- 그 외의 도구인 경우: $\text{Det}_{\text{parts}}$가 찾은 **clevis** 부위와의 $\text{IoU}$를 확인한다.
- 프레임 내의 모든 탐지된 도구($bbox_{tool}$)가 위 조건을 만족하여 $\text{select\_cnt}$가 탐지된 도구의 총수와 일치할 때만 해당 이미지를 PseudoDataset에 포함시켜 다음 라운드 학습에 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Endoscopic Vision Challenge 2023 데이터셋
- **평가 지표**: $\text{mAP}@[.5:.05:0.95]$ (평균 정밀도)
- **비교 방식**: 반복 학습 횟수(Iteration)에 따른 성능 변화 측정

### 정량적 결과

초기 Pseudo-label만 사용했을 때와 비교하여, 반복적인 필터링 과정을 거칠수록 성능이 지속적으로 향상되는 양상을 보였다.

| Iteration | 0 | 1 | 2 | 3 | 4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP (%)** | 4.3 | 10.9 | 13.5 | 14.7 | 15.7 |

실험 결과, 반복 횟수가 증가함에 따라 $\text{mAP}$가 $4.3\%$에서 $15.7\%$까지 상승하였으며, 이는 제안된 다단계 필터링 전략이 Pseudo-label의 품질을 효과적으로 개선하여 모델의 Localization 능력을 향상시켰음을 입증한다.

## 🧠 Insights & Discussion

본 논문은 수술 도구 Localization 분야에서 가장 큰 병목 현상인 '데이터 어노테이션 비용' 문제를 약한 감독 학습(Weakly Supervised Learning)으로 해결하려 했다는 점에서 강점이 있다. 특히 도구의 공통 부품을 탐지하는 모델을 앵커로 활용하여 Pseudo-label을 정제하는 아이디어는 실용적인 접근 방식이다.

다만, 다음과 같은 한계점과 논의사항이 존재한다:

1. **성능의 절대적 수치**: 최종 $\text{mAP}$가 $15.7\%$로, 일반적인 Object Detection 태스크에 비해 매우 낮은 수치이다. 이는 약한 감독 학습의 한계일 수 있으나, 실무 적용 가능성을 위해서는 추가적인 성능 향상이 필요해 보인다.
2. **가정의 위험성**: 도구가 좌측에서 우측 순으로 배치된다는 가정은 단순화된 가설이며, 도구가 복잡하게 겹치거나 배치 순서가 바뀔 경우 초기 라벨 할당 단계에서 심각한 오류가 발생할 가능성이 크다.
3. **SpecialList의 기준**: 특정 도구는 tip을, 나머지는 clevis를 기준으로 필터링하는 기준이 명확히 어떤 근거로 설정되었는지 논문에 구체적으로 명시되지 않았다.

## 📌 TL;DR

이 논문은 수술 도구의 카테고리 정보만을 활용해 위치를 찾아내는 **WS-YOLO** 프레임워크를 제안한다. 도구 공통 부품 탐지 모델을 이용해 초기 Pseudo-label을 만들고, $\text{IoU}$ 기반의 다단계 필터링을 통해 라벨을 정제하며 학습함으로써, 수동 어노테이션 없이도 Localization 성능을 점진적으로 향상($4.3\% \rightarrow 15.7\% \text{ mAP}$)시켰다. 이 연구는 데이터 구축 비용이 매우 높은 의료 영상 분야에서 효율적인 학습 파이프라인을 구축하는 데 기여할 가능성이 있다.
