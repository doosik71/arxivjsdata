# Adaptive Objectness for Object Tracking

Pengpeng Liang, Chunyuan Liao, Xue Mei, and Haibin Ling (2015)

## 🧩 Problem to Solve

시각적 객체 추적(Visual Object Tracking)에서 가장 빈번하게 발생하는 문제 중 하나는 추적기가 타겟을 놓치고 배경의 유사한 영역으로 이동하는 Drifting 현상이다. 기존의 많은 연구들이 타겟의 지역적 구조, 외형, 형태 등 정교한 관측 모델(Observation Model)을 구축하여 이를 해결하려 했으나, "추적 대상은 반드시 객체(Object)여야 하며 비객체(Non-object)가 아니어야 한다"는 매우 단순하고 강력한 사전 지식(Prior Knowledge)은 거의 활용되지 않았다.

최근 제안된 Objectness 측정 방식은 이미지 윈도우가 객체를 포함하고 있을 가능성을 수치화하여 이러한 사전 지식을 모델링할 수 있는 좋은 수단이 된다. 그러나 기존의 Objectness 측정 방식은 다양한 환경과 일반적인 객체를 처리하도록 설계된 Generic한 특성을 가지므로, 특정 추적 시퀀스의 타겟과 환경에 최적화되지 않았다는 한계가 있다. 따라서 본 논문의 목표는 기존의 빠른 Objectness 알고리즘인 BING을 기반으로, 개별 추적 작업에 맞게 적응형으로 학습되는 Adaptive Objectness(ADOBING)를 제안하고 이를 통해 추적 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 일반적인 Objectness 측정치를 특정 추적 대상에 맞게 최적화하는 Transfer Learning 기법을 적용하는 것이다.

1. **ADOBING 제안**: BING(Binarized Normed Gradients)이라는 매우 빠른 Objectness 알고리즘을 베이스로 삼고, 이를 Adaptive SVM을 통해 특정 추적 타겟과 배경에 맞게 조정하여 ADOBING을 구축하였다.
2. **범용적 통합 전략**: 특정 추적기에 종속되지 않고, 기존 추적기의 신뢰도(Confidence) 점수와 ADOBING의 Objectness 점수를 선형 결합(Linear Combination)하는 단순하고 범용적인 방식으로 성능을 개선하였다.
3. **광범위한 검증**: 최신 벤치마크인 CVPR2013과 PTB(Princeton Tracking Benchmark)에서 7가지의 서로 다른 최상위 추적기들을 대상으로 실험하여, ADOBING이 일관되게 성능을 향상시킴을 증명하였다.

## 📎 Related Works

### Visual Object Tracking & Saliency
시각적 Saliency(돌출도)를 추적에 접목하려는 시도가 있었으며, 특히 생물학적 영감을 받은 알고리즘이나 abrupt motion 상황에서 salient region을 검색하여 타겟을 재발견하는 연구들이 존재한다. 하지만 이러한 연구들은 추적 대상이 기본적으로 '객체'여야 한다는 Prior를 명시적으로 모델링하지는 않았다.

### Objectness
Objectness는 이미지 윈도우가 전체 객체를 포함하고 있을 확률을 추정하는 개념이다. 초기 연구들은 다중 스케일 Saliency, 색상 대비, 에지 밀도 등을 사용하여 모델링하였으나 계산 비용이 매우 높았다. 이후 제안된 BING 알고리즘은 Binarized Normed Gradients 특징량을 사용하여 300fps라는 매우 빠른 속도로 Objectness를 계산할 수 있게 하여 실시간 추적 적용의 가능성을 열었다.

### Model Transfer/Adaptation
소스 도메인에서 학습된 지식을 타겟 도메인에 적용하는 Transfer Learning 기법이 SVM 분류기에 적용되어 왔다. 특히 정규화 항(Regularization term)을 수정하여 기존 모델의 지식을 유지하면서 새로운 데이터에 적응시키는 $\ell_1$-regularized linear SVM 등의 연구가 본 논문의 Adaptive SVM 설계에 영감을 주었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 프레임워크는 크게 두 단계로 구성된다. 첫째, 추적 시작 시 첫 번째 프레임의 타겟 정보를 이용하여 Adaptive SVM을 통해 ADOBING 모델을 학습한다. 둘째, 추적 과정에서 베이스 추적기의 추적 신뢰도와 ADOBING의 Objectness 점수를 융합하여 최종 타겟 위치를 결정한다.

### 2. Object-adaptive Objectness (ADOBING) 학습
#### BING의 기초
BING은 이미지 윈도우를 $8 \times 8$ 크기로 리사이징한 후 Normed Gradients를 사용한다. 학습 단계에서는 선형 SVM을 통해 가중치 벡터 $\hat{w} \in \mathbb{R}^{64}$를 학습하며, 테스트 단계에서는 이를 이진 벡터들의 가중 합으로 근사하여 비트 연산(BITWISE, POPCNT)을 통해 매우 빠르게 연산한다.

#### Adaptive SVM 학습 과정
ADOBING은 기존 BING의 가중치 $\hat{w}$를 기반으로, 현재 추적 시퀀스의 데이터 $\mathcal{D} = \{x_i, y_i\}_{i=1}^N$에 최적화된 새로운 가중치 $w$를 학습한다. 이때 기존 모델과의 차이를 제한하면서 분류 오차를 최소화하기 위해 다음과 같은 목적 함수를 사용한다.

$$\min_{w} \|w - \hat{w}\|_1 + C \sum_{i=1}^{N} (\max(0, 1 - y_i w^T x_i))^2$$

여기서 $\|w - \hat{w}\|_1$은 기존 BING 모델로부터의 편차를 제어하는 정규화 항이며, $C$는 정규화 가중치이다.

#### 최적화 알고리즘: Coordinate Descent
위 식은 미분 불가능한 항을 포함하므로, 1차원 Newton direction을 사용하는 Coordinate Descent 알고리즘으로 해결한다. 각 차원 $j$에 대해 다음과 같은 하위 문제(Sub-problem)를 순차적으로 푼다.

$$\min_{z} g_j(z) = |w_{j}^{(k,j)} - \hat{w}_j + z| - |w_{j}^{(k,j)} - \hat{w}_j| + L_j(z; w^{(k,j)}) - L_j(0; w^{(k,j)})$$

여기서 $L_j$는 손실 함수 항이다. 실제 계산 시에는 $L_j$의 2차 근사(second-order approximation)를 사용하여 닫힌 형태(closed-form)의 해 $d$를 구하고, Line search를 통해 충분한 감소 조건(sufficient decrease condition)을 만족하는 최종 업데이트 값을 결정한다.

### 3. 추적기로의 통합 (Encoding)
#### 학습 샘플 생성
첫 번째 프레임에서 슬라이딩 윈도우 방식으로 샘플을 생성한다. Ground Truth와 겹침(overlap) 정도가 임계값보다 높으면 Positive, 그렇지 않으면 Negative로 레이블링한다. 첫 프레임만 사용하는 이유는 추적 중 오염된 샘플이 들어오는 것을 방지하고, Generic한 특성과 Specific한 특성 사이의 균형을 맞춰 과적합(Overfitting)을 막기 위함이다.

#### 신뢰도 융합 (Confidence Fusion)
베이스 추적기 $T$가 제공하는 신뢰도 $f_T(c)$와 ADOBING이 제공하는 Objectness 점수 $f_O(c)$를 다음과 같이 선형 결합하여 최종 신뢰도 $f_{OT}(c)$를 산출한다.

$$f_{OT}(c_i) = f_T(c_i) + \lambda f_O(c_i)$$

여기서 $\lambda$는 상수로, 본 논문에서는 $\lambda = 0.1$을 사용하였다. 모든 베이스 추적기의 결과값은 정규화(Normalize) 과정을 거쳐 결합된다.

## 📊 Results

### 실험 설정
- **벤치마크**: CVPR2013 (50 sequences), Princeton Tracking Benchmark (PTB, 100 sequences).
- **베이스 추적기**: BSBT, Frag, MIL, OAB, SemiT, Struck, TGPR 등 총 7종.
- **지표**: Center Location Error (CLE), Precision (at 20px), Success Plot의 AUC (Area Under Curve).
- **파라미터**: $C = 0.01$, $\lambda = 0.1$.

### 주요 결과
1. **전반적인 성능 향상**: CVPR2013 벤치마크에서 7종의 모든 베이스 추적기가 ADOBING과 결합했을 때 AUC 값이 일관되게 상승하였다 (표 1 참조). 특히 ADOBING-enhanced TGPR은 기존의 최상위 성능을 경신하는 결과를 보였다.
2. **BING vs ADOBING**: 일반 BING을 사용한 경우보다 ADOBING을 사용했을 때 성능 향상 폭이 더 컸으며, BING의 경우 일부 추적기(Struck)에서는 오히려 성능이 하락하는 불안정성을 보였으나 ADOBING은 안정적으로 성능을 높였다.
3. **속성별 분석 (Attribute Analysis)**: 11가지 도전적인 요소 중 Out-of-View(OV) 상황에서 가장 큰 성능 향상이 나타났다. 이는 타겟이 화면 밖으로 나갈 때 발생하는 부분적인 객체(partial objects)들이 낮은 Objectness 점수를 가지므로, ADOBING이 이러한 가짜 타겟으로의 Drifting을 효과적으로 억제했기 때문으로 분석된다.
4. **PTB 벤치마크**: RGBD 데이터셋인 PTB에서도 모든 베이스 추적기의 Success Rate가 ADOBING 결합 후 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 추적 알고리즘의 내부 구조를 수정하지 않고도, "객체성(Objectness)"이라는 단순한 Prior를 추가하는 것만으로 성능을 높일 수 있음을 보여주었다. 특히 Adaptive SVM을 통해 Generic한 모델을 특정 타겟에 맞게 빠르게 전이 학습시킨 점이 주효했다. 이는 복잡한 딥러닝 모델 없이도 도메인 적응(Domain Adaptation)을 통해 실시간 성능과 정확도를 동시에 잡을 수 있음을 시사한다.

### 한계 및 비판적 논의
- **첫 프레임 의존성**: 학습 샘플을 첫 프레임에서만 추출하므로, 첫 프레임의 품질이 매우 중요하다. 또한 추적 과정 중 타겟의 외형이 급격하게 변하는 경우, 첫 프레임에서 학습된 Objectness 모델이 더 이상 유효하지 않을 수 있다.
- **단순 결합 방식**: 신뢰도 점수를 단순히 선형 결합($\lambda$)한 것은 구현이 쉽고 범용적이지만, 추적기마다 점수의 분포와 의미가 다르므로 최적의 $\lambda$ 값이 모든 추적기에 동일하게 적용될 수 있는지에 대한 의문이 남는다.
- **실패 사례**: scale variation이나 illumination variation이 극심한 경우 여전히 실패하는 모습이 관찰되었다. 이는 Objectness가 '객체인지 아닌지'는 판별할 수 있지만, '어떤 객체인지' 또는 '정확히 어디인지'를 정밀하게 짚어내는 능력은 부족하기 때문이다.

## 📌 TL;DR

본 논문은 시각적 객체 추적 시 타겟이 항상 '객체'여야 한다는 사전 지식을 활용하기 위해, 빠른 Objectness 알고리즘인 BING을 특정 타겟에 맞게 최적화한 **ADOBING**을 제안하였다. Adaptive SVM을 통해 첫 프레임에서 타겟 전용 Objectness 모델을 학습하고, 이를 기존 추적기의 신뢰도 점수와 선형 결합함으로써 Drifting 현상을 억제하였다. 7종의 SOTA 추적기에 적용하여 CVPR2013 및 PTB 벤치마크에서 일관된 성능 향상을 입증했으며, 특히 타겟이 화면 밖으로 나가는 상황(Out-of-View)에서 탁월한 효과를 보였다. 이 연구는 단순한 Prior의 통합이 실질적인 추적 성능 향상으로 이어질 수 있음을 보여주며, 향후 다양한 추적 알고리즘의 보조 지표로 활용될 가능성이 높다.