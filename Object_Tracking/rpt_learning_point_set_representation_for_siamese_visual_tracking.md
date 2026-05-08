# RPT: Learning Point Set Representation for Siamese Visual Tracking

Ziang Ma, Linyuan Wang, Haitao Zhang, Wei Lu, and Jun Yin (2020)

## 🧩 Problem to Solve

본 논문은 시각적 추적(Visual Tracking)에서 대상의 상태를 정확하게 추정(Target State Estimation)하는 문제에 집중한다. 기존의 많은 추적기들은 대상의 위치와 크기를 표현하기 위해 Bounding Box(경계 상자) 표현 방식을 사용해 왔다. 그러나 Bounding Box는 대상의 공간적 범위를 매우 거칠게(coarse) 제공하며, 대상의 기하학적 변형(Geometric Transformation)을 모델링하는 능력이 부족하여 국소화 정확도(Localization Accuracy)를 심각하게 제한한다는 문제가 있다.

따라서 본 연구의 목표는 Bounding Box보다 더 세밀한(fine-grained) 표현 방식인 **Point Set Representation(점 집합 표현)**을 도입하여, 대상의 상태를 더 정밀하게 추정하고 모델링하는 효율적인 시각적 추적 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대상을 단순히 하나의 상자로 정의하는 대신, 대상의 의미적·기하학적으로 중요한 지점들을 나타내는 **대표 점들의 집합(Representative Points)**으로 표현하는 것이다.

1. **Point Set Representation 도입**: 대상의 공간적 범위와 주요 지점을 나타내는 점 집합을 학습함으로써, Bounding Box의 거친 인코딩 한계를 극복하고 더 정밀한 국소화와 외형 모델링을 가능하게 한다.
2. **Multi-level Aggregation 전략**: 계층적 컨볼루션 레이어(Hierarchical Convolutional Layers)의 특징들을 융합하여 대상의 상세한 구조 정보를 획득하고, 방해 요소(Distractors)에 대한 판별력을 높인다.
3. **병렬 구조의 서브넷 설계**: 오프라인으로 학습된 점 집합 기반의 '대상 추정 서브넷(Target Estimation Subnet)'과 온라인으로 학습되어 강건성을 높이는 '온라인 분류 서브넷(Online Classification Subnet)'을 병렬로 구성하였다.

## 📎 Related Works

기존의 시각적 추적 연구는 크게 전경-배경 분류 작업과 대상 상태 추정 작업으로 나뉜다.

* **대상 상태 추정 방식**:
  * **Anchor-based (예: SiamRPN 시리즈)**: 미리 정의된 Anchor Box를 사용하여 대상을 회귀하지만, 이는 일반화 성능과 효율성을 저해하는 요소가 된다.
  * **Anchor-free (예: SiamFC++, SiamBAN)**: 사전 지식 없이 캔디데이트 위치에서 경계선까지의 상대적 오프셋(Relative Offsets)을 예측하는 방식을 사용한다.
  * **정밀 표현 방식**: LDES는 회전된 Bounding Box를, SiamMask나 D3S는 세그멘테이션 마스크(Segmentation Mask)를 활용하여 더 상세한 정보를 표현하려 했다. 하지만 마스크 방식은 계산 비용이 높거나 대규모 픽셀 단위 주석 데이터셋이 필요하다는 제약이 있다.

**RPT의 차별점**: RPT는 Bounding Box의 효율성과 세그멘테이션 마스크의 정밀함 사이의 절충안으로 Point Set Representation을 제안한다. 이는 적은 수의 점만으로도 대상의 기하학적 구조를 충분히 표현할 수 있어 효율적이면서도 정확하다.

## 🛠️ Methodology

### 전체 시스템 구조

RPT 프레임워크는 공유 백본 네트워크(Shared Backbone)와 두 개의 병렬 서브넷으로 구성된다. 백본으로는 ResNet-50을 사용하며, 마지막 세 개의 잔차 블록(Residual Blocks)에서 계층적 특징을 추출한다.

### 1. Point Set Representation을 이용한 대상 추정

대상 템플릿 $\mathbf{z}$와 탐색 영역 $\mathbf{x}$로부터 추출된 특징 맵에 대해 depth-wise cross correlation을 수행하여 상관 맵(Correlation Map) $g_l$을 생성한다.
$$g_l(\mathbf{z}, \mathbf{x}) = \phi_l(\mathbf{z}) * \phi_l(\mathbf{x})$$

이후 **RP Head**가 다음과 같이 작동한다:

* **점 집합 초기화**: 상관 맵의 각 위치 $(i, j)$를 대상으로 하는 $n$개의 샘플 점 $R$을 균일하게 초기화한다.
    $$R = \{(x_k, y_k)\}_{k=1}^n, \quad x_k=i, y_k=j$$
* **점 정밀화(Refinement)**: 회귀 헤드가 예측한 오프셋 $\{\Delta x_k, \Delta y_k\}$를 통해 초기 점들을 업데이트한다.
    $$R^r = \{(x_k + \Delta x_k, y_k + \Delta y_k)\}_{k=1}^n$$
    이 과정에서 **Deformable Convolution**을 사용하여 다양한 기하학적 변형(크기, 비율, 회전)을 모델링한다. 본 논문에서는 $n=9$로 설정하였다.
* **Pseudo Box 생성 및 학습**: 점 집합 표현을 Bounding Box 주석으로 학습시키기 위해, 정제된 점들의 최솟값과 최댓값을 이용해 가상의 상자인 Pseudo Box $B_p$를 생성한다.
    $$B_p = (\min\{x_k + \Delta x_k\}, \min\{y_k + \Delta y_k\}, \max\{x_k + \Delta x_k\}, \max\{y_k + \Delta y_k\})$$
    학습 시에는 Pseudo Box와 Ground-truth 간의 IoU를 이용한 회귀 손실과, 전경-배경 분류를 위한 Focal Loss를 사용한다.

### 2. 판별적 온라인 분류 (Discriminative Online Classification)

오프라인 학습만으로는 유사한 방해 요소(Distractor)를 구분하기 어렵기 때문에, 경량화된 2층 FCN 기반의 온라인 분류기를 추가한다.

* **레이블링**: 기존의 가우시안 함수 대신, 학습된 대표 점 집합(Point Set)으로부터의 평균 거리 편차를 이용하여 타겟 존재 확률을 레이블링한다.
* **최종 점수 융합**: 오프라인 분류 결과 $f_{offline}$과 온라인 분류 결과 $f_{online}$을 가중 합산하여 최종 응답 맵을 생성한다.
    $$f_l = \alpha f_{l, online} + (1-\alpha) f_{l, offline}$$

### 3. 다단계 집계 (Multi-level Aggregation)

서로 다른 레이어의 특징(저수준의 세밀한 정보 + 고수준의 의미적 정보)을 융합한다.

* **분류(Classification)**: 각 레벨의 응답 맵을 학습 가능한 가중치 $w_l$을 이용해 픽셀 단위로 합산한다.
    $$R_{cls-all} = \sum_{l=3}^5 w_l * f_l$$
* **추정(Estimation)**: 각 헤드에서 예측된 점 집합들을 모두 합쳐 더 조밀하고 상세한 구조를 가진 Dense Point Set을 구성한다.
    $$R_{est-all} = \bigcup_{l=3}^5 R_l$$

## 📊 Results

### 실험 설정

* **데이터셋**: OTB2015, VOT2018, VOT2019, GOT-10k
* **지표**: Precision, Success rate (AUC), Average Overlap (AO), EAO, Accuracy, Robustness
* **구현**: ResNet-50 백본, GeForce GTX 1080Ti GPU, PyTorch 사용. 20 FPS 이상의 속도로 작동.

### 주요 결과

1. **OTB2015**: AUC 0.715, Precision 0.936을 기록하며 기존 SOTA인 SiamR-CNN(AUC 0.701)과 SiamRPN++(Precision 0.914)를 상회하였다.
2. **GOT-10k**: AO score 0.624를 달성하여 경쟁력 있는 성능을 보였다.
3. **VOT2018/2019**: VOT2018에서 가장 높은 Robustness와 EAO를 기록했으며, VOT2019에서도 모든 지표에서 타 SOTA 모델들을 앞서는 성능(EAO 0.417)을 보였다.
4. **Ablation Study**: Baseline(SiamFC++ 기반) 대비 Point Set Representation(PSR) 추가 시 AUC가 2.5% 상승했으며, 온라인 분류(OC)와 다단계 집계(MLA)를 순차적으로 추가했을 때 각각 0.9%, 0.6%의 추가 상승이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 대상의 상태 표현을 '상자'에서 '점 집합'으로 전환함으로써, 연산 효율성을 유지하면서도 정밀한 국소화를 달성할 수 있음을 입증하였다. 특히 Deformable Convolution을 통해 점들의 위치를 유연하게 조정함으로써 대상의 변형에 강건하게 대응한 점이 주효했다.

**강점 및 한계**:

* **강점**: 점 집합이라는 유연한 표현 방식을 통해 Bounding Box의 단순함과 Segmentation Mask의 복잡함 사이에서 최적의 균형점을 찾았다. 또한 온라인 학습을 병렬로 배치하여 시아미즈 네트워크의 고질적인 문제인 Distractor 취약성을 효과적으로 해결하였다.
* **한계 및 논의**: 본 모델은 오프라인으로 학습된 임베딩 공간에 의존하며, Pseudo Box를 통해 간접적으로 학습하는 방식이므로 실제 점들의 배치와 Bounding Box 간의 완벽한 일치 여부에 대한 심층적인 분석은 부족하다. 또한, VOT2020의 마스크 기반 평가를 위해 D3S와 결합하는 방식을 취했는데, 이는 RPT 자체만으로는 픽셀 단위 마스크를 생성할 수 없음을 의미한다.

## 📌 TL;DR

이 논문은 시각적 추적에서 대상의 상태를 더 정밀하게 추정하기 위해 **Point Set Representation(대표 점 집합 표현)**을 제안한다. 정교한 점들의 배치를 학습하고 이를 다단계 특징 융합 및 온라인 분류기와 결합함으로써, 실시간 속도(20+ FPS)를 유지하면서도 OTB, VOT 등 주요 벤치마크에서 SOTA 성능을 달성하였다. 이 연구는 향후 추적기들이 단순한 상자 형태를 넘어 대상의 기하학적 구조를 어떻게 효율적으로 모델링할 수 있을지에 대한 중요한 방향성을 제시한다.
