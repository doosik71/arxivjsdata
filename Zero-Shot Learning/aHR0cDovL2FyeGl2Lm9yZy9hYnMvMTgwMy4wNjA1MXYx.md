# Deep Multiple Instance Learning for Zero-shot Image Tagging

Shafin Rahman and Salman Khan (2018)

## 🧩 Problem to Solve

본 논문은 이미지 내에 존재하는 여러 개의 객체나 개념을 인식하는 **Multi-label Zero-shot Image Tagging (ZST)** 문제를 해결하고자 한다. 기존의 Zero-shot Learning (ZSL) 모델들은 대부분 이미지당 하나의 unseen label(학습 시 보지 못한 라벨)만을 예측하는 단일 라벨 분류에 집중되어 있었으며, 여러 개의 unseen objects가 동시에 존재하는 실제 환경으로 확장하는 데 한계가 있었다.

이미지 태깅 문제의 핵심은 객체가 이미지의 국소적인 영역(localized region)에 존재할 수도 있고, 전체적인 장면 정보(holistic scene information)를 통해 추론될 수도 있다는 점이다. 따라서 단순한 전역 특징(global feature)만으로는 모든 가능한 라벨을 표현할 수 없으며, 다양한 스케일과 방향, 조명 조건 하에서 국소적/전역적 세부 사항을 모두 탐색해야 하는 도전적인 과제이다. 본 연구의 목표는 이러한 국소적/전역적 정보를 모두 활용하여 seen 및 unseen 라벨을 동시에 예측할 수 있는 end-to-end 학습 가능한 deep MIL(Multiple Instance Learning) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep MIL 프레임워크를 Zero-shot Tagging에 통합하여, 별도의 오프라인 절차 없이 이미지 내의 국소적 특징과 전역적 특징을 모두 추출하고 이를 semantic embedding 공간으로 매핑**하는 것이다. 주요 기여 사항은 다음과 같다.

1. **End-to-End Deep MIL Framework**: multi-label image tagging을 위해 conventional 및 zero-shot 설정 모두에서 작동하는 최초의 진정한 end-to-end deep MIL 프레임워크를 제안한다.
2. **Integrated Bag Generation**: Selective Search나 EdgeBoxes와 같은 외부의 오프라인 절차 없이, 네트워크 자체적으로 MIL을 위한 instance bag을 생성하도록 설계하였다.
3. **Open Vocabulary 확장성**: 테스트 단계에서 semantic embedding 벡터만 제공된다면, 사전 정보가 없는 어떤 수의 새로운 태그(novel tags)에 대해서도 확장이 가능하다.
4. **Weakly Supervised Localization**: 학습 과정에서 bounding box에 대한 정답(ground-truth) 없이도, 예측된 라벨에 대한 bounding box를 생성할 수 있는 능력을 갖추었다.

## 📎 Related Works

**Zero-shot Learning (ZSL)** 분야의 기존 연구들은 주로 도메인 적응, 클래스 속성 연관성 등에 집중하였으나, 이미지 하나에 여러 라벨을 할당해야 하는 실생활의 multi-label 설정에는 대응하지 못하는 한계가 있었다.

**Image Tagging with Deep MIL** 연구들에서는 주로 두 가지 방식이 사용되었다. 첫 번째는 Selective Search 등을 통해 오프라인으로 패치를 생성하고 특징을 추출하는 방식인데, 이는 외부 절차에 의존하므로 end-to-end 학습이 불가능하며 타겟 데이터셋에 맞춰 튜닝될 수 없다는 단점이 있다. 두 번째는 네트워크 내부의 활성화 맵 등을 통해 bag을 생성하는 방식이지만, 이는 국소적인 이미지 세부 사항(localized details)을 충분히 반영하지 못해 비주얼적으로 두드러지지 않는(non-salient) 개념을 인식하는 데 실패하는 경우가 많다.

본 논문은 Faster R-CNN의 Region Proposal Network (RPN)를 활용하여 국소적 특징을 추출함과 동시에 전역 특징을 함께 포함하는 bag을 구성함으로써, 기존의 오프라인 의존성 문제와 국소 정보 부족 문제를 동시에 해결하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 시스템은 크게 **Bag Generation** 부분과 **MIL Network** 부분으로 구성된다.

1. **Bag Generation**: Faster R-CNN 구조를 기반으로 하며, ResNet-50을 backbone으로 사용하여 RPN을 통해 $n$개의 ROI(Region of Interest) 제안을 생성한다. 여기에 이미지 전체를 대표하는 global image ROI($x_{s0}$)를 추가하여 총 $n+1$개의 인스턴스를 가진 bag을 구성한다.
2. **MIL Network**: 생성된 ROI들을 ROI-Pooling 및 Dense layer를 거쳐 $D$차원의 특징 벡터 $F_s = [f_{s0}, \dots, f_{sn}] \in \mathbb{R}^{D \times (n+1)}$로 변환한다.

### 학습 절차 및 주요 방정식

학습 시에는 seen tags($S$)에 대해서만 최적화를 진행한다. 네트워크의 마지막 FC layer의 가중치 $W$는 학습되지 않는 고정된 semantic embedding(GloVe vectors) $W = [v_1, \dots, v_S] \in \mathbb{R}^{d \times S}$를 사용한다.

각 인스턴스에 대한 예측 점수 $P_s$는 다음과 같이 계산된다.
$$P_s = W^T F'_s$$
여기서 $F'_s$는 이전 FC layer들을 통해 $d$차원으로 투영된 특징 벡터이다.

최종적으로 bag의 multi-label 예측 점수 $o_s$는 개별 인스턴스 점수들의 global pooling(max 또는 mean)을 통해 결정된다.

- **Max Pooling**: $o_s = \max(p_{s0}, p_{s1}, \dots, p_{sn})$
- **Mean Pooling**: $o_s = \frac{1}{n+1} \sum_{j=0}^{n} p_{sj}$

### 손실 함수 (Loss Function)

본 논문은 positive tag($y$)와 negative tag($y'$) 간의 점수 차이를 최대화하는 rank-based loss를 사용한다.
$$L_{tag}(o_s, y_s) = \frac{1}{|y_s||S \setminus y_s|} \sum_{y' \in \{S \setminus y_s\}} \sum_{y \in y_s} \log(1 + \exp(o_{y'} - o_y))$$
전체 학습 이미지 $M$개에 대한 총 손실 함수는 다음과 같다.
$$L = \arg \min_{\Theta} \frac{1}{M} \sum_{s=1}^{M} L_{tag}(o_s, y_s)$$

### 추론 및 Zero-shot 확장

테스트 시에는 고정 가중치 $W$를 seen 및 unseen 태그를 모두 포함한 $W' = [v_1, \dots, v_S, v_{S+1}, \dots, v_{S+U}] \in \mathbb{R}^{d \times C}$로 교체한다. 이를 통해 학습 시 보지 못한 unseen 태그에 대해서도 동일한 메커니즘으로 점수를 계산하고 상위 $K$개의 태그를 할당한다.

## 📊 Results

### 실험 설정

- **데이터셋**: NUS-WIDE (81개의 unseen tags와 924개의 seen tags 사용).
- **지표**: Precision (P), Recall (R), F1-score, 그리고 Mean image Average Precision (MiAP)를 측정하였다.
- **비교 대상**: ConSE, Fast0Tag, 그리고 전역 특징만 사용하는 Baseline 모델.

### 주요 결과

1. **태깅 성능**: Conventional, Zero-shot, Generalized zero-shot 모든 태스크에서 제안 방법이 타 모델보다 월등한 성능을 보였다. 특히 Fast0Tag와 비교했을 때 MiAP 지표에서 유의미한 향상을 보였다.
2. **Ablation Study**:
   - Bag size가 커질수록 Mean pooling의 성능이 향상되는 경향을 보였으며, 이는 NUS-WIDE 데이터셋의 어노테이션 노이즈를 평균화하여 완화하는 효과가 있기 때문이다.
   - Max pooling은 상대적으로 작은 bag size에서 더 나은 성능을 보였다.
3. **Zero-shot Recognition (ZSR)**: CUB 데이터셋 실험 결과, 동일한 설정 하에서 state-of-the-art 모델들보다 높은 Top-1 Accuracy를 달성하였다.

## 🧠 Insights & Discussion

**MIL의 효용성**: 분석 결과, 'fish', 'bird', 'bike'와 같이 이미지의 일부 영역에 집중된 태그들은 MIL을 통한 국소 특징 추출이 필수적임을 확인하였다. 반면 'beach', 'sunset'과 같은 전역적 장면 태그들은 global feature가 더 유리하며, 제안 모델은 이 두 가지를 모두 bag에 포함함으로써 상호 보완적인 예측이 가능하게 하였다.

**약점 및 한계**: 전역적인 정보만으로 판단해야 하는 태그의 경우, 때때로 Fast0Tag와 같은 전역 기반 모델보다 성능이 낮게 나오는 경우가 관찰되었다. 또한, NUS-WIDE 데이터셋 자체의 극심한 노이즈로 인해 매우 많은 수의 unseen tags(4,084개)를 처리할 때는 전반적인 성능이 급격히 저하되는 한계가 있었다.

**비판적 해석**: 본 연구는 bounding box를 생성하는 부가적인 기능을 제공하지만, 이에 대한 정량적 평가가 이루어지지 않았다. 이는 NUS-WIDE 데이터셋에 localization ground-truth가 없기 때문인데, 실제 객체 검출 성능이 어느 정도인지 검증하기 위해서는 별도의 데이터셋이나 외부 평가 지표가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 **end-to-end로 학습 가능한 Deep MIL 프레임워크를 제안하여 Multi-label Zero-shot Image Tagging 문제를 해결**하였다. RPN을 통해 국소 영역과 전역 영역을 동시에 캡처하는 instance bag을 생성하고, 이를 semantic embedding 공간에 매핑함으로써 학습하지 않은 라벨에 대해서도 효과적인 태깅이 가능함을 입증하였다. 이 연구는 향후 open-vocabulary 기반의 이미지 태깅 및 weakly supervised object localization 연구에 중요한 기초를 제공할 것으로 기대된다.
