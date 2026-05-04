# BiSeg: Simultaneous Instance Segmentation and Semantic Segmentation with Fully Convolutional Networks

Viet-Quoc Pham, Satoshi Ito, Tatsuo Kozakaya (2017)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 **Instance Segmentation**의 정확도를 높이는 것이다. Instance Segmentation은 객체 검출(Object Detection)의 수준인 바운딩 박스 단위를 넘어, 픽셀 단위로 객체를 정밀하게 국지화(Localization)해야 하므로 높은 수준의 검출 정확도와 정교한 세그멘테이션 능력을 동시에 요구한다.

기존의 방식들은 마스크 제안(Mask Proposal) 단계에서 시맨틱 카테고리 정보를 활용하지 못하거나, 여러 단계의 네트워크를 거쳐야 하므로 추론 속도가 느리고 정확도가 떨어진다는 한계가 있었다. 또한, 이미지 내의 '객체(Things)'뿐만 아니라 '배경 요소(Stuff, 예: 하늘, 잔디)'까지 함께 처리할 수 있는 통합된 프레임워크의 필요성이 제기되었다. 따라서 본 논문의 목표는 Fully Convolutional Networks (FCNs)를 기반으로 시맨틱 세그멘테이션과 인스턴스 세그멘테이션을 동시에 수행하는 효율적인 end-to-end 솔루션을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **시맨틱 세그멘테이션(Semantic Segmentation) 결과를 인스턴스 세그멘테이션(Instance Segmentation)을 위한 사전 확률(Prior)로 사용하는 베이지안 추론(Bayesian Inference)** 방식의 도입이다.

주요 기여 사항은 다음과 같다:

1. **베이지안 프레임워크 적용**: 시맨틱 세그멘테이션 결과($P(c|X)$)를 Prior로, Position-sensitive score maps를 Likelihood로 사용하여 인스턴스 세그멘테이션의 Posterior를 예측한다.
2. **Fused Position-Sensitive Score Maps**: 기존 FCIS에서 사용하던 단일 스케일의 스코어 맵을 확장하여, 서로 다른 스케일(Conv3, Conv5)과 서로 다른 파티션 모드($k \times k$)의 스코어 맵들을 융합함으로써 파라미터 의존성을 줄이고 강건함(Robustness)을 높였다.
3. **End-to-End FCN 구조**: 모든 연산이 픽셀 단위로 수행되는 Fully Convolutional 구조를 유지하여 FCN의 장점을 그대로 계승하면서도 PASCAL VOC 데이터셋에서 최첨단(State-of-the-art) 성능을 달성하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 한계를 지적한다:

- **DeepMask, SharpMask, Instance FCN**: 마스크 제안 생성에 집중하며 시맨틱 카테고리에 둔감하다. 분류를 위해 Fast R-CNN과 같은 별도의 다운스트림 네트워크가 필요하며, 이미지 피라미드 스캔 방식을 사용하여 추론 속도가 느리다.
- **MNC (Multi-task Network Cascade)**: 박스 제안, 마스크 회귀, 카테고리 분류의 3단계 구조를 가지며 end-to-end 솔루션을 제공하지만, 단계별 구조로 인해 FCIS보다 속도가 느리다.
- **FCIS (Fully Convolutional Instance-aware Semantic Segmentation)**: Position-sensitive score maps를 통해 마스크 예측과 분류를 동시에 수행하는 FCN 기반의 효율적인 모델이다. BiSeg는 이 FCIS의 구조를 기반으로 하되, 시맨틱 세그멘테이션과의 결합 및 스코어 맵 융합을 통해 성능을 개선하였다.

## 🛠️ Methodology

### 전체 시스템 구조

BiSeg는 ResNet-101을 백본으로 하며, 공유된 컨볼루션 특징 맵(Convolutional Feature Maps)을 기반으로 네 가지 서브 네트워크가 병렬적으로 동작한다:

1. **Region Proposal Network (RPN)**: 후보 영역(ROI)을 생성한다.
2. **Bounding Box Regression**: 생성된 ROI를 정밀하게 조정한다.
3. **Semantic Segmentation Sub-network**: 이미지 전체의 시맨틱 세그멘테이션 확률을 추론한다.
4. **Instance Segmentation Sub-network**: 각 ROI에 대한 인스턴스 세그멘테이션의 Likelihood를 추정한다.

### Fusion of Position-Sensitive Score Maps

인스턴스 세그멘테이션의 정확도를 높이기 위해, 본 논문은 서로 다른 스케일과 파티션 모드의 스코어 맵을 융합한다.

- **구성**:
  - 첫 번째 세트: $\text{conv5}$ 레이어에서 생성되며, $k_1 \times k_1$ (기본값 $7 \times 7$) 파티션을 가진 $2k_1^2 \times (C+1)$ 개의 스코어 맵을 생성한다.
  - 두 번째 세트: $\text{conv3}$ 레이어에서 생성되며, $k_2 \times k_2$ (기본값 $9 \times 9$) 파티션을 가진 $2k_2^2 \times (C+1)$ 개의 스코어 맵을 생성한다.
- **절차**: 각 ROI에 대해 두 세트의 스코어 맵에서 각각 Likelihood 맵을 어셈블(Assemble)한다. 이후 첫 번째 세트의 맵을 $2\times$ 업샘플링하여 두 번째 세트의 맵과 합산함으로써 최종 Likelihood 맵 $L$을 도출한다.

### 베이지안 추론 (Bayesian Inference)

인스턴스 세그멘테이션 확률 $I$를 계산하기 위해 다음과 같은 베이지안 수식을 적용한다. 입력 이미지 $X$가 주어졌을 때, 카테고리 $c$와 $k$번째 ROI에 대한 인스턴스 세그멘테이션의 사후 확률 $P(I^{ck}|X)$는 다음과 같이 정의된다:

$$P(I^{ck}|X) = P(I^{ck}, c|X) = P(I^{ck}|c, X)P(c|X)$$

여기서 각 항의 역할은 다음과 같다:

- $P(c|X)$: **Prior**. 시맨틱 세그멘테이션 서브 네트워크에서 예측한 시맨틱 확률 맵 $S$를 사용하여 근사한다.
- $P(I^{ck}|c, X)$: **Likelihood**. 앞서 설명한 Fused Position-sensitive score maps를 통해 얻은 ROI inside likelihood 맵 $L$을 사용하여 근사한다.

최종적으로 시맨틱 확률 맵과 인스턴스 Likelihood 맵의 원소별 곱(Element-wise product)을 통해 인스턴스 세그멘테이션 확률을 도출한다.

### 학습 절차 및 손실 함수

모델은 다음과 같은 멀티태스크 손실 함수를 통해 학습된다:

$$L = L_{rpn} + L_{ss} + L_{cls} + L_{mask} + L_{bbox}$$

- $L_{rpn}$: Faster R-CNN의 RPN 손실과 동일하다.
- $L_{ss}$: 픽셀 단위의 Multinomial Cross-entropy 손실을 사용하여 시맨틱 세그멘테이션을 학습한다.
- $L_{cls}, L_{mask}, L_{bbox}$: 각 ROI에 대해 소프트맥스 분류 손실, 이진 교차 엔트로피 마스크 손실, 바운딩 박스 회귀 손실을 적용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL VOC 2012 (Train set 5,623장, Val set 5,732장).
- **평가 지표**: $mAP_r$ (Mean Average Precision over regions)를 사용하며, IoU 임계값을 0.5와 0.7로 설정하여 측정한다.
- **비교 대상**: FCIS, MNC, IIS 및 자체 베이스라인인 Naive Multi-task, Single PS score map BiSeg 등.

### 주요 결과

정량적 결과는 Table 1에 제시되어 있으며, 핵심 수치는 다음과 같다:

- **BiSeg (fused PS score map)**: $mAP_r@0.5 = 67.3\%$, $mAP_r@0.7 = 54.4\%$를 기록하며 FCIS(65.7%, 52.1%) 대비 약 2% 높은 성능 향상을 보였다.
- **Ablation Study (파티션 모드)**: Table 2에 따르면 $(k_1, k_2) = (7, 9)$ 조합이 $(7, 7)$ 조합보다 효과적임을 확인하였다. 다만 $k_2=11$과 같이 너무 큰 값은 과분할(Over-partition)로 인해 성능이 저하되었다.
- **시맨틱 세그멘테이션 성능**: Table 3에서 BiSeg는 Naive Multi-task보다 높은 Mean Accuracy(70.2%)와 Mean IU(60.8%)를 기록하여, 두 태스크의 결합이 상호 보완적으로 작용함을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 시맨틱 세그멘테이션과 인스턴스 세그멘테이션 사이에 강한 상관관계가 있다는 직관을 베이지안 프레임워크로 풀어내어 성공적으로 성능을 향상시켰다. 특히, 별도의 CRF(Conditional Random Field) 없이도 경계선 부분에서 높은 세그멘테이션 품질을 보여주었다는 점이 인상적이다.

**강점 및 한계**:

- **강점**: FCN의 효율성을 유지하면서도 Prior 정보를 활용해 인스턴스 검출의 정밀도를 높였으며, 다중 스케일 맵 융합을 통해 하이퍼파라미터(파티션 크기)에 대한 의존성을 완화하였다.
- **한계 및 가정**: 시맨틱 세그멘테이션 모델의 효과를 명확히 검증하기 위해 모델을 매우 단순하게 유지하였다. 따라서 CRF나 Higher-order potential 같은 고도화된 시맨틱 세그멘테이션 기법을 적용했을 때 인스턴스 세그멘테이션 성능이 얼마나 더 향상될지는 미해결 질문으로 남겨두었다.

**비판적 해석**:
본 논문은 Prior의 도입이 성능 향상의 핵심임을 보였으나, 이는 결국 시맨틱 세그멘테이션의 정확도가 인스턴스 세그멘테이션의 상한선을 결정할 수 있음을 의미한다. 즉, Prior가 잘못 예측될 경우 Posterior 결과에 부정적인 영향을 줄 가능성이 있으며, 이에 대한 분석이나 완화 전략에 대한 논의가 부족한 점이 아쉽다.

## 📌 TL;DR

BiSeg는 시맨틱 세그멘테이션 결과를 Prior로, 융합된 Position-sensitive score maps를 Likelihood로 사용하는 베이지안 추론 기반의 FCN 프레임워크이다. 이를 통해 PASCAL VOC 데이터셋에서 기존 FCIS 대비 약 2% 향상된 $mAP_r$ 성능을 달성하였다. 이 연구는 시맨틱 정보와 인스턴스 정보의 결합이 상호 이득을 준다는 것을 입증하였으며, 향후 실시간 객체 분할 및 군중 계수(Object Counting) 등의 응용 분야에 중요한 기여를 할 가능성이 높다.
