# Real-time Instance Segmentation of Surgical Instruments using Attention and Multi-scale Feature Fusion

Juan Carlos Ángeles-Cerón, Gilberto Ochoa-Ruiz, Leonardo Chang and Sharib Ali (2021)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 최소 침습 수술(Minimally Invasive Surgery, MIS) 환경에서 수술 도구의 정확하고 실시간적인 인스턴스 분할(Instance Segmentation)을 구현하는 것이다. 수술 도구의 정밀한 추적은 외과 의사의 수술 경로 탐색을 돕고 환자의 안전을 증진시키는 데 필수적이다.

그러나 이를 구현하는 데에는 다음과 같은 두 가지 주요 어려움이 존재한다. 첫째, 출혈, 연기, 반사, 조명 변화 및 가림(occlusion)과 같은 복잡하고 동적인 수술 환경으로 인해 모델의 강건성(robustness)을 확보하기 어렵다. 둘째, 높은 정확도를 제공하는 2단계(two-stage) 검출 모델은 추론 속도가 너무 느려 실시간 적용이 불가능하며, 반대로 속도가 빠른 경량 모델은 내시경 이미지의 복잡한 특성을 충분히 잡아내지 못해 정확도가 떨어진다는 상충 관계(trade-off)가 존재한다. 따라서 본 논문의 목표는 실시간 성능(Real-time performance)과 높은 분할 정확도를 동시에 달성하는 강건한 단일 단계(single-stage) 인스턴스 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 실시간 인스턴스 분할 모델인 YOLACT++를 베이스라인으로 채택하고, 여기에 도메인 특화 최적화 기법들을 단계적으로 결합하여 정확도와 강건성을 극대화하는 것이다. 주요 기여 사항은 다음과 같다.

- **Attention Mechanism 도입**: ResNet-101 백본과 Feature Pyramid Network(FPN) 출력단에 CCAM(Criss-cross Attention Module)과 CBAM(Convolutional Block Attention Module)을 적용하여 전역적 문맥 정보와 중요한 특징을 효과적으로 추출한다.
- **Multi-scale Feature Fusion (MSFF) 설계**: 백본의 서로 다른 스케일에서 추출된 특징들을 융합하여 저수준(low-level)과 고수준(high-level) 세만틱 정보를 동시에 유지하는 구조를 제안한다.
- **앵커 박스(Anchor Box) 최적화**: Differential Evolution search 알고리즘을 사용하여 수술 도구의 크기와 비율에 최적화된 앵커 스케일 및 비율을 설정함으로써 검출율(recall)을 높였다.
- **도메인 특화 데이터 증강(Data Augmentation)**: 수술 환경의 특성을 반영한 광도 왜곡(photometric distortions) 및 아핀 변환(affine transformations)을 적용하여 일반화 성능을 향상시켰다.

## 📎 Related Works

기존의 수술 도구 분할 연구들은 주로 UNet, LinkNet, TernausNet과 같은 시맨틱 분할 모델이나, Mask R-CNN과 같은 2단계 인스턴스 분할 모델에 의존하였다. 특히 ROBUST-MIS 챌린지 참여자들의 대부분은 Mask R-CNN 기반 모델을 사용하였는데, 이들은 강건성 측면에서는 준수한 성능을 보였으나 추론 속도가 평균 5 FPS 수준으로 매우 느려 실제 임상 환경의 실시간 요구사항을 충족하지 못했다.

반면, YOLACT와 같은 단일 단계 모델은 마스크 프로토타입(prototype masks)과 선형 조합 계수를 예측하는 단순화된 구조 덕분에 매우 빠르지만, Mask R-CNN에 비해 정확도 격차가 컸다. 본 연구는 이러한 격차를 줄이기 위해 YOLACT++의 구조적 장점을 유지하면서, 어텐션 모듈과 다중 스케일 융합을 통해 의료 영상 특유의 복잡한 환경에 대응하고자 하였다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 프레임워크는 ResNet-101 백본, MSFF 모듈, Attention 모듈, 그리고 FPN으로 구성된다. 입력 이미지는 백본을 통해 특징 맵을 생성하고, MSFF를 통해 다양한 스케일의 정보가 융합된다. 이후 어텐션 모듈이 특징 표현을 정교화하며, 최종적으로 ProtoNet(마스크 프로토타입 생성)과 Prediction Head(박스 및 계수 예측)를 통해 인스턴스 마스크를 생성한다.

### 주요 구성 요소 및 상세 설명

**1. Multi-scale Feature Fusion (MSFF)**
다양한 해상도의 특징 맵 $F_s$ ($s$는 스케일 레벨)를 통합하여 고해상도 표현을 유지하면서도 풍부한 문맥 정보를 얻기 위해 사용된다.

- 모든 스케일의 특징 맵 $F_s$를 전치 합성곱(transposed convolution)을 통해 가장 높은 해상도로 업샘플링하여 $\hat{F}_s$를 생성한다.
- 업샘플링된 모든 특징 맵을 연결(concatenation)한 후 합성곱 층을 통과시켜 통합 특징 맵 $F_{MS}$를 생성한다:
$$F_{MS} = \text{conv}([\hat{F}_0, \hat{F}_1, \hat{F}_2, \hat{F}_3, \hat{F}_4])$$
- 다시 $F_{MS}$를 각 $\hat{F}_s$와 연결하고 합성곱을 적용하여 최종 융합된 특징 맵 $F_A$를 생성한다.

**2. Attention Mechanisms**
연산 오버헤드를 최소화하기 위해 CCAM과 CBAM을 사용한다.

- **CCAM (Criss-cross Attention Module)**: 이미지의 가로-세로 방향으로 십자 형태의 어텐션을 적용하여 전역적인 의존성을 효율적으로 모델링한다.
- **CBAM (Convolutional Block Attention Module)**: 채널 어텐션과 공간 어텐션을 순차적으로 적용하여 '무엇이' 중요한지와 '어디가' 중요한지를 동시에 학습한다.

**3. 학습 목표 및 손실 함수**
모델은 다음 네 가지 손실 함수의 가중 합으로 학습된다:

- 분류 손실 ($\mathcal{L}_{cls}$): Softmax Cross Entropy 사용 (도구 vs 배경).
- 박스 회귀 손실 ($\mathcal{L}_{box}$): Smooth-L1 손실 사용.
- 마스크 손실 ($\mathcal{L}_{mask}$): 예측 마스크와 정답 마스크 간의 픽셀 단위 Binary Cross Entropy 사용.
- 시맨틱 분할 손실 ($\mathcal{L}_{sem}$): 특징 풍부성을 위해 훈련 중에만 평가되는 보조 손실.
최종 손실 함수는 $\mathcal{L} = 1\mathcal{L}_{cls} + 1.5\mathcal{L}_{box} + 6.125\mathcal{L}_{mask} + 1\mathcal{L}_{sem}$으로 정의된다.

**4. 앵커 최적화**
이미지 크기 $600 \times 600$ 픽셀에 맞춰 Differential Evolution search 알고리즘을 통해 최적의 앵커 스케일 $[0.435, 0.502, 0.578, 0.664, 0.762]$과 비율 $[0.267, 0.554, 1.0, 1.804, 3.746]$을 도출하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: ROBUST-MIS 데이터셋 (10,040 프레임). 특히 일반화 성능 검증을 위해 학습 시 보지 못한 데이터인 Stage 3 테스트 셋을 주 평가 대상으로 사용하였다.
- **평가 지표**: 면적 기반의 $\text{MI\_DSC}$ (Multi-instance Dice)와 거리 기반의 $\text{MI\_NSD}$ (Multi-instance Normalized Surface Dice)를 사용하였으며, 추론 속도는 FPS(Frames Per Second)로 측정하였다.
- **비교 대상**: 2019 ROBUST-MIS 챌린지의 상위 팀들과 베이스라인 YOLACT++ 모델.

### 정량적 결과

가장 성능이 좋은 모델인 `CBAM-Full + Aug + Anch + MS`는 Stage 3에서 $\text{MI\_DSC} = 0.447$, $\text{MI\_NSD} = 0.489$를 기록하여 챌린지 상위 팀(Team www) 대비 $\text{MI\_DSC}$ 기준 약 13.7% 향상된 결과를 보였다.

추론 속도와 정확도의 균형을 고려한 `CBAM-Full + Aug + Anch` 모델의 경우, $\text{MI\_DSC} = 0.425$, $\text{MI\_NSD} = 0.471$의 높은 정확도를 유지하면서도 $69 \text{ FPS}$라는 매우 빠른 속도를 달성하였다. 이는 기존 Mask R-CNN 기반 모델들보다 약 13.8배 빠른 속도이다.

### 정성적 결과

최악의 성능을 보인 프레임(worst-case samples)을 분석한 결과, 베이스라인 모델은 연기, 혈액, 모션 블러가 있는 상황에서 검출 실패가 잦았으나, 제안된 기법들을 단계적으로 적용함에 따라 다음과 같은 개선이 확인되었다.

- **Attention & Augmentation**: 필드 가장자리의 작은 도구 및 투명한 도구의 검출 능력이 향상되었다.
- **Anchor Optimization**: 길고 수직인 도구의 검출 및 분할 정확도가 크게 개선되었다.
- **MSFF**: 저조도 환경의 도구 및 반사가 심한 영역에서의 강건성이 증가하였다.

## 🧠 Insights & Discussion

본 연구는 단일 단계 모델의 속도적 이점을 유지하면서도, 어텐션과 다중 스케일 융합을 통해 2단계 모델에 근접하거나 이를 능가하는 정확도를 확보할 수 있음을 입증하였다.

특히 **CBAM과 CCAM의 비교**를 통해, CBAM이 백본과 FPN 모두에 적용되었을 때 더 우수한 성능을 보였음을 확인하였다. 이는 CBAM이 이미 통합된 특징 데이터에서 채널과 공간 정보를 정교하게 재정제(refine)하는 능력이 더 뛰어나기 때문으로 분석된다. 반면 CCAM은 특징이 섞이기 전 단계에서 더 효과적이었다.

또한, 단순한 아키텍처 변경뿐만 아니라 **도메인 특화 앵커 최적화와 데이터 증강**이 실제 의료 영상의 변동성(도구의 다양한 각도와 크기)을 해결하는 데 결정적인 역할을 했음을 알 수 있다. 다만, 가장 정밀한 모델(`+ MS` 포함)의 경우 연산 복잡도 증가로 인해 속도가 $24 \text{ FPS}$로 저하되는 한계가 있었으나, 이 역시 기존 SOTA 대비 4.8배 빠른 수준이다.

## 📌 TL;DR

본 논문은 YOLACT++를 기반으로 **CBAM 어텐션, 다중 스케일 특징 융합(MSFF), 앵커 최적화, 도메인 특화 증강**을 결합하여 수술 도구의 실시간 인스턴스 분할 프레임워크를 제안하였다. 이 모델은 ROBUST-MIS 챌린지의 기존 최고 성능 모델들을 정량적으로 앞질렀으며, 특히 $69 \text{ FPS}$의 속도를 유지하면서도 높은 강건성을 확보하여 실제 임상 적용 가능성을 높였다. 이 연구는 향후 실시간 수술 내비게이션 및 로봇 수술 시스템의 핵심 모듈로 활용될 가능성이 매우 높다.
