# FGN: Fully Guided Network for Few-Shot Instance Segmentation

Zhibo Fan, Jin-Gang Yu, Zhihao Liang, Jiarong Ou, Changxin Gao, Gui-Song Xia, Yuanqing Li (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Few-Shot Instance Segmentation (FSIS)이다. Instance Segmentation은 이미지 내 객체의 위치를 찾고(Localization), 클래스를 분류하며(Classification), 픽셀 단위의 마스크를 생성(Mask Estimation)하는 복합적인 작업이다. 일반적으로 이를 수행하기 위해서는 방대한 양의 레이블링된 데이터가 필요하지만, 실제 환경에서는 모든 클래스에 대해 충분한 데이터를 확보하는 것이 불가능하거나 비용이 매우 많이 든다.

Few-Shot Learning (FSL) 패러다임을 Instance Segmentation에 접목하면, 소수의 샘플(Support Set)만으로도 새로운 클래스(Novel Classes)에 대해 추론할 수 있는 일반화 성능을 얻을 수 있다. 하지만 Instance Segmentation 네트워크는 구조가 매우 복잡하여, 단순히 기존의 FSL 방식을 적용하는 것만으로는 충분한 가이드(Guidance)를 제공하기 어렵다는 문제가 있다. 따라서 본 논문의 목표는 Support Set의 정보를 네트워크의 각 핵심 구성 요소에 효과적으로 전달하여, 새로운 클래스에 대해서도 높은 정밀도의 세그멘테이션 결과를 얻는 Fully Guided Network (FGN)를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mask R-CNN의 서로 다른 구성 요소들이 수행하는 역할이 다르므로, 각각에 최적화된 **차별화된 가이드 메커니즘(Different Guidance Mechanisms)**을 도입해야 한다는 것이다. 기존 연구들이 네트워크의 특정 지점에 단일 가이드 모듈을 추가하여 모든 구성 요소가 동일한 가이드를 공유하게 했던 한계를 극복하고, 다음과 같은 세 가지 맞춤형 가이드 모듈을 제안한다.

1.  **Attention-Guided RPN (AG-RPN)**: RPN이 클래스 불가지론적(Class-agnostic) 제안만 생성하는 것이 아니라, Support Set의 정보를 통해 특정 클래스에 집중한 클래스 인식(Class-aware) 제안을 생성하도록 유도한다.
2.  **Relation-Guided Detector (RG-DET)**: Support Set의 특징과 RoI(Region of Interest) 특징을 명시적으로 비교함으로써, 클래스 간 일반화 성능을 높이고 배경 클래스를 효과적으로 배제한다.
3.  **Attention-Guided FCN (AG-FCN)**: 마스크 생성 단계에서 Support Set의 어텐션 정보를 활용하여 세밀한 픽셀 단위 가이드를 제공한다.

## 📎 Related Works

### 관련 연구 및 한계
1.  **Instance Segmentation**: Mask R-CNN과 같은 Fully-supervised 방식은 높은 성능을 보이지만, 데이터 의존성이 매우 높다는 한계가 있다.
2.  **Few-Shot Learning (Classification/Semantic Segmentation)**: 이미지 분류나 시맨틱 세그멘테이션 분야에서는 이미 많은 FSL 연구가 진행되었으나, Instance Segmentation은 객체의 위치 찾기와 마스크 생성이 동시에 이루어져야 하므로 구조적 복잡도가 훨씬 높다.
3.  **기존 FSIS 접근 방식**: 
    *   **Siamese MRCNN**: 백본 네트워크를 Siamese 구조로 만들어 첫 단계에서 가이드를 제공하지만, 이후의 모든 컴포넌트(RPN, Heads)가 동일한 가이드를 공유해야 하는 제약이 있다.
    *   **Meta R-CNN**: 두 번째 단계의 시작 부분에서 클래스 어텐션 벡터를 통해 특징 맵을 재가중치(Reweight)하지만, 첫 번째 단계인 RPN의 가이드를 완전히 무시한다는 단점이 있다.

본 논문의 FGN은 이러한 단일 지점 가이드 방식에서 벗어나, RPN부터 Mask head까지 전체 파이프라인에 걸쳐 최적화된 가이드를 개별적으로 주입함으로써 'Full Guidance'를 달성했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
FGN은 Mask R-CNN을 기반으로 하며, Support Set $S$와 Query Image $x$를 공유 백본 $\phi$ (ResNet101)를 통해 각각 특징 맵 $F^k_n$과 $Y$로 인코딩한다. 이후 AG-RPN, RG-DET, AG-FCN의 세 단계로 가이드가 적용된다.

### 1. Attention-Guided RPN (AG-RPN)
RPN이 새로운 클래스의 객체를 놓치지 않고 정확하게 제안(Proposal)을 생성하도록 돕는다.
- **특징 추출**: Support Set의 특징 맵 $F^k_n \in \mathbb{R}^{H \times W \times C}$에 Global Average Pooling (GAP)을 적용하고 클래스별로 평균을 내어 클래스 어텐션 벡터 $a_n$을 생성한다.
  $$a_n = \frac{1}{K} \sum_{k=1}^{K} \text{GAP}(F^k_n), \quad n=1, \dots, N$$
- **특징 가중치 적용**: Query 이미지의 특징 맵 $Y$에 $a_n$을 요소별 곱(Element-wise multiplication)하여 클래스별로 가중치가 적용된 특징 맵 $\tilde{Y}_n$을 생성한다.
  $$\tilde{Y}_n = Y \otimes a_n, \quad n=1, \dots, N$$
- **제안 생성**: 각 $\tilde{Y}_n$을 독립적인 RPN에 입력하여 제안을 생성한 후, 이를 통합하여 최종 class-aware proposal을 도출한다.

### 2. Relation-Guided Detector (RG-DET)
분류(Classification)와 바운딩 박스 회귀(Bbox Regression) 단계에서 Support Set과의 명시적 비교를 수행한다.
- **명시적 비교**: Relation Network의 개념을 도입하여, RoI 정렬 특징 $z_j$와 Support Set의 정렬 특징 $\hat{F}_n$을 결합(Concatenate)한 후 MLP(Multi-Layer Perceptron)에 통과시킨다.
- **배경 제거 메커니즘**: 단순히 클래스를 분류하는 것이 아니라, 각 클래스에 대해 매칭 점수 $c^+_i$와 배경 점수 $c^-_i$의 쌍을 출력한다. 최종 배경 점수 $c_{N+1}$은 가장 매칭 점수가 높은 클래스의 배경 점수 $c^-_{i^*}$를 사용하여 결정함으로써 배경을 효과적으로 배제한다.
  $$c = (c_1, \dots, c_N, c_{N+1}) \text{ where } c_i = c^+_i, c_{N+1} = c^-_{i^*} \text{ and } i^* = \arg \max_i \{c^+_i\}$$

### 3. Attention-Guided FCN (AG-FCN)
마지막 마스크 생성 단계에서 정밀한 가이드를 제공한다.
- **Masked Pooling**: Support Set의 특징 $\hat{F}^k_n$에 대해 바이너리 마스크 $\hat{m}^k_n$ 영역 내에서만 풀링을 수행하는 Masked Pooling을 적용하여 클래스 어텐션 벡터 $b_n$을 생성한다.
- **가이드 적용**: 학습 시에는 Ground Truth 클래스를, 테스트 시에는 분류 점수가 가장 높은 클래스의 벡터 $b_{n^*}$를 선택하여 Query RoI 특징 $z_j$에 적용한다.
  $$\tilde{z}_j = z_j \otimes b_{n^*}$$

### 학습 절차 (Training Strategy)
학습은 두 단계로 진행된다.
1.  **1단계**: Base 클래스 데이터 $D_{base}$만을 사용하여 기본 모델을 학습시킨다.
2.  **2단계**: Base 클래스와 Novel 클래스를 모두 포함한 데이터 $(C_{base} \cup C_{novel})$를 사용하여 모델을 미세 조정(Fine-tuning)한다. 이때 에피소드 기반 샘플링을 통해 Few-Shot 상황을 시뮬레이션하여 학습한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Microsoft COCO 2017, PASCAL VOC 2012.
- **설정**: 
    - **COCO2VOC**: COCO의 60개 클래스를 Base, VOC의 20개 클래스를 Novel로 설정한 교차 데이터셋 설정.
    - **VOC2VOC**: VOC 내에서 15개 Base, 5개 Novel 클래스로 설정.
- **태스크**: 1-way 1-shot, 3-way 1-shot, 3-way 3-shot.
- **지표**: $\text{mAP}_{50}$ (Mean Average Precision).

### 주요 결과
- **성능 우위**: COCO2VOC 설정에서 FGN은 Siamese MRCNN 및 Meta R-CNN보다 월등한 성능을 보였다. 특히 3-way 3-shot 세그멘테이션에서 기존 모델들보다 크게 높은 $\text{mAP}_{50}$를 기록했다 (Table 1).
- **검출 vs 세그멘테이션**: 모든 모델에서 Detection 성능보다 Segmentation 성능이 낮게 나타났는데, 이는 FSIS가 단순한 객체 검출의 확장이 아니라 훨씬 더 어려운 작업임을 시사한다.
- **일반화 능력**: 데이터셋 규모가 작은 VOC2VOC 설정에서도 FGN이 가장 우수한 성능을 유지하며 강건함을 증명했다 (Table 3).

### 절제 연구 (Ablation Study)
- **전체 가이드의 효과**: AG-RPN, RG-DET, AG-FCN 중 하나라도 제거했을 때 $\text{mAP}_{50}$가 하락하는 것을 확인하여, 세 가지 모듈이 모두 성능 향상에 기여함을 입증했다 (Table 4).
- **개별 모듈 검증**: 제안한 AG-RPN이 기존 RPN이나 타 모델의 방식보다 높은 AR(Average Recall)을 기록했으며, AG-FCN 역시 기존의 시맨틱 세그멘테이션 가이드 방식보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Few-Shot Instance Segmentation을 해결하기 위해 네트워크의 모든 주요 지점에서 서로 다른 특성의 가이드를 제공하는 것이 필수적임을 보여주었다. 

**강점 및 통찰**:
- **구조적 최적화**: RPN에는 '관심 영역 집중', Detector에는 '명시적 비교 및 배경 배제', FCN에는 '마스크 기반 어텐션'이라는 각 단계의 목적에 맞는 최적의 가이드를 설계한 점이 매우 논리적이다.
- **문제의 본질 파악**: 단순히 모델의 파라미터를 조정하는 것이 아니라, FSIS의 본질적인 어려움인 '클래스 간 일반화'와 '배경 제거' 문제를 구체적인 모듈(RG-DET)로 해결하려 노력했다.

**한계 및 비판적 해석**:
- **절대적 성능의 한계**: SOTA 성능을 달성했음에도 불구하고, Fully-supervised 방식에 비해 절대적인 성능 수치가 매우 낮다. 이는 FSIS 자체가 매우 도전적인 과제임을 뜻하기도 하지만, 제안된 가이드 방식만으로는 여전히 부족함이 있음을 의미한다.
- **배경 제거의 단순성**: RG-DET에서 배경 점수를 결정할 때 가장 높은 매칭 점수를 가진 클래스에 의존하는 방식이 모든 상황에서 최적인지는 의문이며, 더 정교한 배경 분리 메커니즘이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 Mask R-CNN의 각 구성 요소(RPN, Detector, FCN)에 최적화된 세 가지 개별 가이드 메커니즘(AG-RPN, RG-DET, AG-FCN)을 도입한 **Fully Guided Network (FGN)**를 제안한다. 이를 통해 소수의 샘플만으로도 새로운 클래스의 객체를 정확히 검출하고 세그멘테이션할 수 있게 하였으며, 실험적으로 기존 FSIS 모델들을 능가하는 성능을 입증했다. 이 연구는 향후 적은 데이터만으로 고성능 인스턴스 세그멘테이션을 구현하려는 연구에 중요한 가이드라인을 제공한다.