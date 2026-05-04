# SiamVGG: Visual Tracking using Deeper Siamese Networks

Yuhong Li, Xiaofan Zhang, Deming Chen (2022)

## 🧩 Problem to Solve

본 논문은 시각적 객체 추적(Visual Object Tracking) 분야에서 고정밀도(High Accuracy)와 신뢰할 수 있는 실시간 성능(Real-time Performance)을 동시에 달성하고자 한다. 기존의 DNN 기반 추적기들은 높은 연산 복잡도로 인해 처리 시간이 길어져 실시간 성능을 보장하기 어려웠다. 특히, 최근 주목받는 Siamese Network 기반의 추적기들은 온라인 업데이트가 필요 없어 속도는 빠르지만, 주로 AlexNet과 같은 비교적 얕은 네트워크를 백본(Backbone)으로 사용하여 객체 식별 능력(Discrimination Capability)이 부족하다는 한계가 있다. 결과적으로 유사한 배경이나 객체가 존재할 때 쉽게 혼동되어 추적 성능이 저하되는 문제가 발생한다. 따라서 본 연구의 목표는 더 깊은 네트워크 구조를 도입하여 식별 능력을 강화하면서도, 실시간성을 유지하는 SiamVGG 추적기를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Siamese Network의 구조적 이점을 유지하면서, 더 강력한 특징 추출 능력을 가진 VGG-16 네트워크를 백본으로 채택하여 객체 식별 능력을 극대화하는 것이다.

주요 기여 사항은 다음과 같다.

1. **VGG-16 기반의 Siamese 구조 설계**: AlexNet보다 깊은 VGG-16을 백본으로 사용하여 더 상세한 세만틱 특징(Semantic Features)을 캡처함으로써 식별력을 높였다.
2. **패딩(Padding) 제거**: 일반적인 CNN에서 사용하는 패딩 연산이 Siamese Network의 cross-correlation 결과인 score map에 노이즈를 유발하여 성능을 저하시킨다는 점을 발견하고, 이를 제거한 설계를 적용하였다.
3. **효율적인 학습 전략**: ILSVRC와 Youtube-BB 데이터셋을 적절한 비율(1:5)로 혼합하여 학습함으로써 과적합을 방지하고 일반화 성능을 높였다.

## 📎 Related Works

### 1. CNN 기반 특징 추출 방식

Deep-SRDCF, C-COT, ECO와 같은 추적기들은 CNN을 통해 고차원 세만틱 특징을 추출하고 이를 Correlation Filter와 결합하여 높은 정확도를 달성하였다. 그러나 이러한 방식들은 입력 프레임마다 CNN의 모든 레이어를 통과해야 하므로 연산량이 매우 많으며, 특히 입력 이미지의 크기가 커질 경우 실시간 성능을 확보하기 어렵다는 한계가 있다.

### 2. Siamese Network 기반 추적 방식

SiamFC는 추적 문제를 유사도 학습(Similarity Learning) 문제로 정의하여, 템플릿 이미지(Exemplary image)와 검색 영역(Search image) 간의 유사도를 계산함으로써 온라인 업데이트 없이 빠르게 타겟을 찾는다. 이후 EAST, DSiam, SA-Siam, SiamRPN 등이 제안되었으나, 대부분 AlexNet을 백본으로 사용하고 있어 복잡한 환경에서의 식별 능력이 부족하다는 공통적인 한계가 존재한다.

## 🛠️ Methodology

### 전체 시스템 구조

SiamVGG는 두 개의 입력(템플릿 이미지 $z$와 검색 이미지 $x$)을 동일한 가중치를 공유하는 CNN 백본 $\phi$에 통과시킨 후, 두 특징 맵 간의 cross-correlation을 통해 타겟의 위치를 예측하는 구조이다.

### 주요 구성 요소 및 절차

1. **백본 네트워크 (Modified VGG-16)**: VGG-16의 처음 10개 레이어를 사용하며, 마지막 레이어는 $1 \times 1$ 커널을 사용하여 출력 채널을 조정한다. 특히 패딩을 제거하여 score map의 품질을 높였으며, 이로 인해 발생하는 특징 맵의 크기 감소 문제를 해결하기 위해 네트워크 깊이를 정교하게 설정하였다.
2. **유사도 계산 (Cross-Correlation)**: 두 입력 이미지의 특징 맵 $\phi(z)$와 $\phi(x)$ 사이의 유사도를 계산하는 함수 $f(z, x)$는 다음과 같이 정의된다.
   $$f(z, x) = \phi(z) * \phi(x)$$
   여기서 $*$는 cross-correlation 연산을 의미하며, 기존 SiamFC에 있던 bias 항($b_1$)은 성능에 기여도가 낮아 제거하여 구조를 경량화하였다.
3. **추론 절차**: 예측된 score map에서 최대 점수를 가진 지점을 타겟의 위치로 결정한다. 또한, 스케일 변화에 대응하기 위해 단일 추론 시 여러 스케일의 이미지를 배치(mini-batch) 형태로 처리한다.

### 학습 방법

1. **데이터셋 및 정답 생성**: ILSVRC와 Youtube-BB 데이터셋을 1:5 비율로 사용한다. 템플릿 이미지는 $127 \times 127$, 검색 이미지는 $255 \times 255$ 크기로 구성한다.
2. **Ground Truth (GT) 생성**: Score map의 중심 $c$로부터 Manhattan 거리 기반의 반경 $R$ 내에 타겟이 위치하면 $+1$, 그렇지 않으면 $-1$로 설정한다.
   $$y[u] = \begin{cases} +1 & \text{if } k\|u-c\| \le R \\ -1 & \text{otherwise} \end{cases}$$
3. **손실 함수 (Loss Function)**: 예측 score map $x$와 정답 $y$ 사이의 오차를 줄이기 위해 SoftMargin loss를 사용한다.
   $$\text{loss}(x, y) = \frac{\sum_i \log(1 + \exp(-y[i] \times x[i]))}{n}$$
   여기서 $n$은 score map의 전체 요소 수이다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB-2013/50/100, VOT 2015/2016/2017.
- **지표**: OTB에서는 AUC(Area Under Curve)를, VOT에서는 EAO(Expected Average Overlap), Overlap, Failures를 사용한다.
- **환경**: Nvidia GTX 1080Ti GPU 사용.

### 주요 결과

1. **정량적 성능**:
   - **OTB-100**: AUC 0.654를 달성하여 기존 Siamese 기반 추적기들과 경쟁력 있는 성능을 보였다.
   - **VOT2015/2016**: EAO 기준 각각 0.373, 0.351을 기록하며 1위를 차지하였다.
   - **VOT2017**: Overlap에서는 1위, EAO에서는 3위를 기록하였다. 특히 실시간 추적기들 간의 비교에서 SiamRPN 대비 EAO가 13% 향상된 0.275를 달성하였다.
2. **실시간성**: GTX 1080Ti 환경에서 최대 50 FPS의 처리 속도를 보여, 실시간 응용 분야에 충분히 적용 가능한 수준임을 입증하였다.
3. **정성적 분석**: Figure 1을 통해 SiamFC가 배경 노이즈에 취약하여 score map이 불분명한 반면, SiamVGG는 타겟 위치에서 매우 뚜렷한 피크(Peak)를 생성하여 식별 능력이 월등히 높음을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 단순히 네트워크를 깊게 만든 것이 아니라, **패딩 제거**라는 실무적인 통찰을 통해 Siamese 구조에서의 score map 품질을 개선하였다. 또한, Batch Normalization(BN) 유무에 따른 ablation study를 통해, 분류 작업에서는 BN이 유리하지만 **추적 작업에서는 BN이 없는 모델(AUC 0.654)이 있는 모델(AUC 0.589)보다 훨씬 높은 성능을 낸다**는 중요한 발견을 하였다.

### 한계 및 논의

- **데이터 의존성**: 대규모 데이터셋(ILSVRC, Youtube-BB)을 통한 오프라인 학습에 의존하므로, 학습 데이터에 포함되지 않은 특수한 환경의 객체에 대한 적응력은 검증되지 않았다.
- **온라인 업데이트 부재**: 실시간성을 위해 온라인 파라미터 업데이트를 포기했기 때문에, 타겟의 외형이 급격하게 변하는 상황(Appearance change)에서의 강건성은 제한적일 수 있다.

## 📌 TL;DR

SiamVGG는 기존 Siamese Network의 약점인 낮은 식별력을 해결하기 위해 **패딩이 제거된 VGG-16 백본**을 도입한 실시간 객체 추적기이다. 오프라인 학습을 통해 높은 정밀도를 확보하면서도 GTX 1080Ti 기준 **50 FPS**라는 빠른 속도를 달성하였으며, 특히 VOT2017 챌린지에서 기존 실시간 추적기들보다 월등한 EAO 성능을 입증하였다. 이 연구는 깊은 네트워크를 Siamese 구조에 효율적으로 통합하는 방법을 제시하여, 향후 실시간 고정밀 추적 시스템 및 IoT 기기 적용 가능성을 높였다.
