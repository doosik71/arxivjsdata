# Automatic Sleep Stage Classification with Cross-modal Self-supervised Features from Deep Brain Signals

Chen Gong, Yue Chen, Yanan Sui, Luming Li (2023)

## 🧩 Problem to Solve

본 연구는 파킨슨병(Parkinson's Disease, PD) 환자에게 이식된 심부 뇌 자극기(Deep Brain Stimulator, DBS)를 통해 기록된 Local Field Potentials(LFP) 신호를 이용하여 수면 단계(Sleep Stage)를 자동으로 분류하는 것을 목표로 한다.

수면 단계의 정확한 진단은 신경 및 정신 질환의 치료에 필수적이며, 특히 폐루프(Closed-loop) DBS 시스템을 구현하기 위해서는 환자의 수면 상태를 실시간으로 감지하여 최적의 자극 파라미터를 설정하는 것이 매우 중요하다. 기존의 수면 단계 분류는 다원수면검사(Polysomnography, PSG)에 의존했으나, 이는 많은 수의 웨어러블 센서로 인해 수면의 질을 저하시키는 문제가 있다. LFP 기반의 분류 방식은 추가적인 센서 없이 일상생활에서의 모니터링이 가능하다는 장점이 있지만, 기존의 머신러닝 기반 분류기들은 주파수 도메인 특징의 일관성이 부족하여 일반화 성능이 낮고, 임상 데이터의 규모가 작아 딥러닝 모델을 학습시키기에 데이터가 불충분하다는 한계가 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 뇌 신호를 "듣는(Listening)" 관점으로 접근하여, 음성 신호(Speech signal)와 LFP 신호가 시간-주파수 도메인에서 유사한 형태를 가진다는 가설 하에 **교차 모달 자가 지도 학습(Cross-modal Self-supervised Learning)**을 적용한 것이다.

대규모 음성 데이터로 사전 학습된 WavLM 모델을 LFP 신호의 특징 추출기로 사용하여, 임상 데이터 부족 문제를 해결하고 LFP 신호 내의 고수준 표현(High-level representation)을 효과적으로 포착하였다. 또한, 전극 채널 간의 중요도를 동적으로 학습하는 Self-attention 메커니즘을 도입하여 피험자마다 다른 전극 위치에 따른 가변성을 극복하고 일반화 성능을 높였다.

## 📎 Related Works

기존의 수면 단계 분류는 주로 PSG를 통해 수행되었으며, LFP를 이용한 시도에서는 인공신경망(ANN), 서포트 벡터 머신(SVM), 랜덤 포레스트(Random Forest) 등이 사용되었다. 이러한 기존 방식들은 주로 LFP의 주파수 도메인 에너지 특징에 기반하였으나, 수면 단계별 주파수 특성이 피험자마다 일관되지 않아 새로운 데이터에 대한 강건성(Robustness)이 부족했다.

최근 딥러닝 기반의 특징 표현 학습이 주목받고 있으나, 뇌 신호 데이터의 특성상 대규모 레이블링 데이터 확보가 어렵다. 이에 본 논문은 레이블이 없는 대규모 데이터를 활용하는 자가 지도 학습(Self-supervised Learning)과 서로 다른 도메인의 지식을 전이하는 전이 학습(Transfer Learning)을 통해 이 문제를 해결하고자 하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 모델은 LFP 신호 입력부터 수면 단계 분류까지 이어지는 end-to-end 구조로 설계되었으며, 크게 **특징 표현(Feature Representation) $\rightarrow$ 셀프 어텐션(Self-attention) $\rightarrow$ 분류(Classification)** 프레임워크로 구성된다.

### 2. 상세 구성 요소 및 절차

**가. 전처리 (Preprocessing)**

- LFP 신호를 5초 단위의 세그먼트로 분할한다.
- 0.5Hz 고역 통과 필터(High-pass filter)를 적용하여 움직임이나 시스템 노이즈를 제거한다.
- WavLM 모델의 입력 규격에 맞추기 위해 샘플링 레이트를 $16,000\text{Hz}$로 설정한다(이는 기존 $500\text{Hz}$ 신호를 32배 가속하는 것과 동일한 효과를 가진다).

**나. 특징 표현 프레임워크 (Feature Representation)**

- 대규모 무감독 음성 데이터로 사전 학습된 **WavLM Large** 모델을 사용한다.
- WavLM은 7층의 CNN 인코더와 24층의 Transformer 인코더로 구성되어 있다.
- 학습 비용을 줄이기 위해 WavLM의 파라미터는 동결(Freeze)시킨 채 특징 추출기로만 사용하며, 각 Transformer 디코더 층의 잠재 출력(Latent output)을 추출하여 총 25개의 특징 텐서(wavLM 0-24)를 획득한다.

**다. 채널 셀프 어텐션 프레임워크 (Self-attention)**

- 8개 채널의 LFP 신호는 전극과 뇌 영역 간의 상대적 위치 정보를 담고 있다. 수술 시 발생하는 위치 가변성을 해결하기 위해 각 채널의 중요도를 학습하는 어텐션 구조를 설계하였다.
- 입력 특징 텐서 $F = (F_i), i=1, 2, \dots, 8$에 대해 선형 변환을 통해 Query($Q$), Key($K$), Value($V$) 벡터를 생성한다.
$$Q_i = W_Q F_i, \quad K_i = W_K F_i, \quad V_i = W_V F_i$$
- 각 채널의 가중치 점수 $\alpha_i$는 다음과 같이 계산된다.
$$\alpha_i = \text{softmax}\left(\frac{Q_i \cdot K_i}{\sqrt{d}}\right)$$
여기서 $d$는 벡터의 차원이다. 최종 출력 $H$는 가중치 합으로 표현된다.
$$H = \sum_{i=1}^{8} \alpha_i \cdot V_i$$

**라. 분류 프레임워크 (Classification)**

- 어텐션 네트워크의 출력을 입력으로 하여 시간적 특성을 추출하는 3층의 1-D CNN(커널 크기 5)을 사용한다.
- CNN의 출력 $o^T$를 선형 매핑하고 합산하여 최종 수면 단계를 결정한다.
$$O = \text{CNN}(H)$$
$$w^T = \text{softmax}(W'_T \cdot o^T)$$
$$S = \sum_T w^T \cdot o^T$$

### 3. 학습 목표 및 손실 함수

수면 단계 데이터의 불균형(Wake : N1 : N2/N3 : REM $\approx 1.5 : 1 : 5.5 : 2.5$)을 해결하기 위해 **Focal Loss**를 사용하였다.
$$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 파킨슨병 환자 12명의 하룻밤 LFP 신호 데이터.
- **분류 대상**: 4단계 수면 상태 (Wake, N1, N2/N3, REM).
- **태스크**:
  - Classification Task: 무작위로 90% 학습, 10% 테스트.
  - Prediction Task: 시간 순서대로 90% 학습, 10% 테스트 (실제 온라인 시나리오 모사).
- **기준선(Benchmark)**: 기존 연구에서 제안된 SVM 기반 모델.

### 2. 주요 결과

- **정량적 결과**:
  - Classification Task: 본 모델의 정확도는 **83.2%**로, Benchmark(65.6%) 대비 크게 향상되었다.
  - Prediction Task: 본 모델의 정확도는 **68.2%**로, Benchmark(58.9%) 대비 향상되었다.
- **단계별 성능**: Wake(89.1%)와 REM(88.8%)에서 가장 높은 성능을 보였으며, N1(75.8%)에서 상대적으로 낮은 성능을 보였다. 이는 N1 단계가 다른 단계들 사이의 전이(Transition) 구간이기 때문으로 분석된다.
- **Ablation Study**:
  - Benchmark (65.6%) $\rightarrow$ CNN + Self-attention (70.7%) $\rightarrow$ End-to-End (83.2%) 순으로 성능이 향상되어, 교차 모달 특징 추출기가 성능 향상에 핵심적인 역할을 했음을 입증하였다.

## 🧠 Insights & Discussion

**1. 저수준 특징의 중요성**
WavLM 모델의 층별 가중치를 분석한 결과, 음성 인식 태스크와 달리 LFP 수면 분류에서는 **얕은 층(Shallow layers)의 가중치가 매우 높게** 나타났다. 이는 LFP 신호에서 수면 단계를 구분 짓는 핵심 요소가 고도의 추상적 의미보다는 생리학적으로 관련된 저수준(Low-level) 특징들에 있음을 시사한다.

**2. 전극 위치와 수면의 기능적 연결**
셀프 어텐션 가중치를 시각화하여 분석한 결과, 가중치가 높은 채널이 실제 치료용 전극(Therapy electrodes) 또는 그 인접 전극과 일치하는 경향이 발견되었다. 이는 파킨슨병 환자의 치료 타겟 영역과 수면 조절 기전 사이에 기능적인 연결성이 존재할 가능성을 보여준다.

**3. 한계점**
본 연구는 12명의 소규모 환자 집단을 대상으로 하였으며, 이식 후 장기간에 따른 LFP 신호의 가변성을 충분히 반영하지 못했다. 향후 장기적인 LFP 수면 데이터셋 구축을 통한 검증이 필요하다.

## 📌 TL;DR

본 논문은 파킨슨병 환자의 심부 뇌 신호(LFP)를 이용하여 수면 단계를 분류하기 위해, **음성 인식 모델인 WavLM을 이용한 교차 모달 전이 학습**을 제안하였다. 뇌 신호를 마치 음성 신호처럼 "듣는" 방식으로 처리함으로써 임상 데이터 부족 문제를 해결하였으며, 셀프 어텐션 메커니즘을 통해 전극 채널의 가변성을 극복하였다. 실험 결과, 기존 SVM 기반 방식보다 월등한 정확도(최대 83.2%)를 달성하였으며, 이는 향후 환자 맞춤형 폐루프 DBS 시스템의 실시간 수면 모니터링 및 치료 전략 수립에 중요한 기초가 될 것으로 기대된다.
