# An Experimental Survey on Correlation Filter-based Tracking

Zhe Chen, Zhibin Hong, and Dacheng Tao (2015)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야의 난제 중 하나인 시각적 객체 추적(Visual Object Tracking)에서 최근 각광받고 있는 Correlation Filter 기반 추적기(Correlation Filter-based Trackers, 이하 CFTs)들의 발전 과정을 체계적으로 분석하고 평가하는 것을 목표로 한다.

시각적 객체 추적은 조명 변화, 가려짐(occlusion), 변형(deformation), 회전 등 다양한 환경적 요인으로 인해 빠른 속도와 강건함(robustness)을 동시에 확보하는 것이 매우 어렵다. 특히 CFTs는 매우 효율적인 연산 속도와 준수한 성능으로 많은 주목을 받았으나, 다양한 CFT 알고리즘들에 대한 통합적인 프레임워크 제시와 대규모 벤치마크를 통한 정밀한 비교 분석이 부족한 상태였다. 따라서 본 연구는 CFTs의 일반적인 구조를 정립하고, 다양한 학습 기법 및 성능 개선 방안을 분석하여 향후 연구 방향을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **일반적 프레임워크 정립**: 다양한 CFTs의 공통적인 작동 원리를 분석하여 일반적인 추적 파이프라인을 공식화하였다.
2. **학습 스킴 분석**: MOSSE, KCF, STC 등 서로 다른 학습 전략(선형, 커널 기반, 문맥 기반)을 심층적으로 분석하고 수학적으로 설명하였다.
3. **대규모 실험적 검증**: OOTB(Online Object Tracking Benchmark) 데이터셋을 사용하여 11개의 CFTs와 29개의 타 추적기들을 대상으로 정량적/정성적 비교 분석을 수행하였다.
4. **성능 개선 요소 도출**: 특징 표현(Feature representation), 스케일 변화 대응(Scale variations), 부분 기반 추적(Part-based tracking), 장기 추적(Long-term tracking) 관점에서 CFTs의 한계와 개선 방안을 논의하였다.

## 📎 Related Works

논문은 객체 추적 방식을 크게 생성 모델(Generative models)과 판별 모델(Discriminative models)로 구분한다. 생성 모델은 최적의 매칭 윈도우를 찾는 방식이며, 판별 모델은 타겟과 배경을 구분하는 법을 학습한다. 최근 연구들에 따르면 배경 정보의 활용이 추적 성능을 높이는 데 유리하므로 판별 모델이 더 경쟁력이 있다고 평가된다.

특히 CFTs의 시초가 된 MOSSE(Minimum Output Sum of Squared Error) 필터는 적응형 학습 스킴을 도입하여 온라인 추적의 효율성을 획기적으로 높였다. 이후 커널 방법을 도입한 CSK, HOG 특징을 결합한 KCF, 컬러 속성을 활용한 CN 등이 제안되었으며, 최근에는 스케일 변화를 해결하려는 DSST, SAMF와 같은 연구들이 등장하며 성능을 끌어올렸다.

## 🛠️ Methodology

### 1. 일반적인 CFT 프레임워크

CFT의 기본 워크플로우는 다음과 같다. 첫 프레임에서 타겟 패치를 통해 필터를 학습하고, 이후 프레임에서는 이전 위치 주변의 패치를 추출하여 검출을 수행한다.

* **검출 단계**: 입력 데이터 $x$와 학습된 필터 $h$의 상관관계(correlation)를 계산한다. 이때 계산 효율을 위해 Discrete Fourier Transform(DFT)을 사용하여 공간 영역의 합성곱(convolution)을 주파수 영역의 원소별 곱셈(element-wise multiplication)으로 대체한다.
    $$x \otimes h = \mathcal{F}^{-1}(\hat{x} \odot \hat{h}^*)$$
    여기서 $\mathcal{F}^{-1}$은 역 푸리에 변환, $\odot$은 원소별 곱셈, $*$은 켤레 복소수를 의미한다. 결과물인 응답 맵(Response map)에서 최댓값을 가진 위치를 타겟의 새로운 위치로 예측한다.
* **학습 단계**: 새로운 타겟 인스턴스 $x'$가 주어졌을 때, 원하는 출력 $y$(일반적으로 가우시안 분포)를 얻기 위한 필터 $\hat{h}^*$를 다음과 같이 계산한다.
    $$\hat{h}^* = \frac{\hat{y}}{\hat{x}'}$$

### 2. 주요 학습 스킴(Training Schemes)

#### A. MOSSE (선형 필터)

MOSSE는 실제 상관관계 출력과 원하는 출력 사이의 오차 제곱합을 최소화하는 필터를 찾는다.
$$\min_{\hat{h}^*} \sum_{i} \|\hat{x}_i \odot \hat{h}^* - \hat{y}_i\|^2$$
최종 솔루션은 다음과 같다.
$$\hat{h}^* = \frac{\sum_i \hat{y}_i \odot \hat{x}_i^*}{\sum_i \hat{x}_i \odot \hat{x}_i^*}$$

#### B. KCF (Kernelized Correlation Filter)

KCF는 Ridge Regression 문제와 Circulant Matrix(순환 행렬)의 성질을 이용하여 커널 트릭을 적용한다.

* **Ridge Regression**: 손실 함수 $L(\cdot)$과 정규화 항 $\lambda \|w\|^2$를 최소화한다.
* **Circulant Matrix**: 타겟 주변의 모든 가능한 이동(translation) 샘플을 효율적으로 학습하기 위해 사용한다. 순환 행렬은 DFT를 통해 대각 행렬로 변환될 수 있어 계산량이 매우 적다.
* **주파수 영역 솔루션**: 커널 행렬의 베이스 벡터를 $k$라고 할 때, 필터 계수 $\alpha$는 다음과 같이 계산된다.
    $$\hat{\alpha} = \frac{\hat{y}}{\hat{k} + \lambda}$$

#### C. STC (Spatio-Temporal Context)

STC는 단순한 외형 학습이 아니라, 객체 주변의 공간-시간적 문맥(context) 정보를 활용하여 우도 분포(likelihood distribution) $\ell(p) = P(p|o)$를 학습한다.

* **핵심 아이디어**: 주변 픽셀 값과 상대적 위치의 관계를 학습하여, 타겟이 중심에 있을 확률을 모델링한다.
* **스케일 추정**: $\ell(p)$의 신뢰도 점수 비율을 통해 스케일을 추정하는 독자적인 방식을 가진다.

### 3. 성능 개선 기법

* **특징 표현**: Raw pixel 대신 HOG(그래디언트 분석)와 Color Names(색상 정보)를 결합하여 조명 변화 및 노이즈에 강건하게 대응한다.
* **스케일 대응**: Scaling Pool(다양한 크기의 윈도우를 샘플링하여 최적의 점수를 찾는 방식)을 도입한 SAMF, DSST 등이 대표적이다.
* **부분 기반 추적(Part-based)**: 타겟을 여러 부분으로 나누어 독립적으로 추적하고 이를 결합함으로써 부분 가려짐(partial occlusion) 문제에 대응한다(예: RPT).
* **장기 추적(Long-term)**: 단기 메모리(CFT)와 장기 메모리(Key points 기반)를 결합한 MUSTer의 구조처럼, 타겟 분실 시 재검출(re-detection) 모듈을 통해 추적을 재개한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: OOTB (51개 및 100개 시퀀스)
* **평가 지표**: OPE(One-Pass Evaluation), TRE(Temporal Robustness), SRE(Spatial Robustness). Precision plot(20px 임계값)과 Success plot(AUC)을 사용한다.
* **비교 대상**: MUSTer, RPT, SAMF, DSST, KCF, MOSSE 등 11개의 CFTs와 MEEM, TLD, Struck 등 28개의 타 추적기.

### 2. 정량적 결과

* **종합 성능**: MUSTer가 대부분의 Success plot에서 가장 높은 AUC를 기록하며 최상의 성능을 보였다.
* **속도 및 정확도**: KCF, MOSSE 등은 매우 빠른 속도(FPS)를 보이지만, MUSTer나 SAMF와 같은 개선된 CFTs는 속도를 일부 희생하더라도 훨씬 높은 정확도(OS, Overlap Score)와 낮은 위치 오차(CLE)를 기록하였다.
* **속성별 분석**:
  * **스케일 변화**: MUSTer, DSST, SAMF가 우수한 성능을 보였다.
  * **빠른 움직임(Fast Motion)**: MEEM이 가장 강건하였다.
  * **가려짐 및 배경 혼란**: CFT 계열 추적기들이 배경 문맥을 효율적으로 구분하여 타 추적기들보다 우수한 성능을 보였다.
  * **저해상도**: MUSTer가 가장 우수하였으며, 부분 기반 추적기인 RPT는 해상도 저하에 민감하게 반응하여 성능이 급감하는 경향을 보였다.

### 3. 정성적 결과

* **KCF**: 고정 윈도우 크기로 인해 스케일 변화가 심한 영상(CarScale 등)에서 박스 크기가 맞지 않는 한계가 드러났다.
* **DSST/SAMF**: 스케일 변화에는 잘 대응하나, 장기 메모리가 없어 타겟을 한 번 놓치면 재포착하지 못하는 한계가 있었다.
* **MUSTer**: 타겟을 놓친 상황(MotorRolling 등)에서도 장기 메모리 컴포넌트를 통해 성공적으로 재포착하는 모습을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 한계

* **강점**: CFTs는 FFT를 이용한 연산 최적화 덕분에 실시간 추적이 가능하며, 특히 배경 정보를 학습에 포함함으로써 판별 능력이 매우 뛰어나다.
* **한계**: 기본적으로 고정된 필터 크기를 사용하므로 스케일 변화에 취약하며, 드리프트(drift) 발생 시 이를 스스로 복구할 수 있는 재검출 메커니즘이 부족하다.

### 2. 비판적 해석 및 논의

본 논문은 CFTs의 발전 방향이 '단순한 필터 학습'에서 '특징 융합 $\rightarrow$ 스케일 대응 $\rightarrow$ 장기 추적'으로 확장되고 있음을 보여준다. 특히 MUSTer의 성과는 단기적인 빠른 추적(CFT)과 장기적인 강건한 복구(Key points)의 결합이 시각적 추적의 핵심 해결책이 될 수 있음을 시사한다. 다만, 성능이 좋은 추적기일수록 연산 속도가 느려지는 경향이 있어, 고성능과 고속을 동시에 달성하는 최적화 연구가 여전히 필요하다.

## 📌 TL;DR

본 논문은 Correlation Filter 기반 추적기(CFTs)의 일반적 프레임워크를 정립하고, 다양한 학습 기법과 개선 방안을 대규모 벤치마크(OOTB)를 통해 분석한 종합 서베이 보고서이다. 실험 결과, 단순 CFT보다는 HOG/컬러 특징을 결합하고 스케일 추정 및 장기 메모리 모델(예: MUSTer)을 도입한 추적기들이 압도적인 강건함을 보였다. 이 연구는 향후 CFT 연구가 단순한 필터 설계를 넘어 효율적인 스케일 추정, 정교한 부분 기반 융합, 그리고 신뢰성 있는 재검출 메커니즘의 결합 방향으로 나아가야 함을 제시한다.
