# End-to-End Automatic Sleep Stage Classification Using Spectral-Temporal Sleep Features

Hyeong-Jin Kim, Minji Lee, and Seong-Whan Lee (2020)

## 🧩 Problem to Solve

수면 장애는 일상생활의 질에 큰 영향을 미치는 신경계 질환 중 하나이다. 수면 장애를 진단하기 위해서는 수면 단계(sleep stages)를 분류하는 과정이 필수적인데, 전통적으로는 전문가가 Polysomnography (PSG) 신호를 수동으로 판독하여 등급을 매기는 방식을 사용한다. 그러나 이러한 수동 분류 방식은 매우 많은 시간과 비용이 소요되며 효율성이 떨어진다는 치명적인 단점이 있다.

이를 해결하기 위해 딥러닝 기반의 자동 수면 단계 분류 기술이 연구되어 왔으나, 기존의 raw signal(원시 신호)만을 이용한 방식들은 여전히 실용적인 수준의 높은 분류 성능을 달성하지 못하고 있다. 따라서 본 논문의 목표는 수면 단계별로 뚜렷하게 나타나는 주파수 특성을 반영한 최적의 spectral-temporal 특징(spectral-temporal features)을 추출하여, 자동 수면 단계 분류의 성능을 향상시키는 end-to-end 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 수면 단계마다 고유하게 나타나는 주파수 대역의 특성을 모델이 직접 학습할 수 있도록, 입력 데이터 단계에서 최적화된 주파수 필터링 신호를 함께 제공하는 것이다. 단순히 raw data를 입력하는 대신, 수면 과학에서 중요하게 다루는 특정 주파수 대역(Delta, Theta, Alpha, Sleep Spindle)을 Bandpass Filter로 추출하여 입력 행렬에 포함함으로써, 모델이 시간적 정보와 주파수 특성을 동시에 효과적으로 학습할 수 있도록 설계하였다.

## 📎 Related Works

기존의 자동 수면 단계 분류 연구들은 주로 딥러닝, 특히 Convolutional Neural Networks (CNN)를 활용해 왔다. 대표적으로 Supratak 등이 제안한 DeepSleepNet은 단일 채널 EEG의 raw signal을 사용하며, 두 개의 CNN 스트림과 Bidirectional LSTM을 결합한 복잡한 구조를 사용하였다. Phan 등은 EEG와 EOG 채널을 함께 사용하며 joint classification과 prediction을 위한 multi-task softmax layer를 도입하였다. 또한 Patanaik과 Yildirim 등은 층을 깊게 쌓은 CNN 모델을 통해 특징 추출 성능을 높이려 시도하였다.

하지만 이러한 기존 연구들은 수면 단계별로 명확한 주파수 특성(예: N3 단계의 Delta wave, N2 단계의 Sleep spindle 등)이 존재함에도 불구하고, 이러한 도메인 지식을 입력 특징으로 직접 활용하지 않았다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 입력 데이터 구성

본 연구에서는 raw EEG 및 EOG 신호를 전처리하여 CNN 모델에 입력하는 end-to-end 구조를 제안한다. 입력 데이터는 총 6개의 행(row)으로 구성된 행렬 형태이며, 각 행은 다음과 같은 신호를 포함한다.

1. Raw EEG signal (Fpz-Cz 채널)
2. Bandpass filtered EEG - Delta 대역 ($0.5 \text{--} 4\text{ Hz}$)
3. Bandpass filtered EEG - Theta 대역 ($4 \text{--} 8\text{ Hz}$)
4. Bandpass filtered EEG - Alpha 대역 ($8 \text{--} 12\text{ Hz}$)
5. Bandpass filtered EEG - Sleep spindle 대역 ($12 \text{--} 15\text{ Hz}$)
6. Raw EOG signal (Horizontal 채널)

비교를 위한 대조군(Control group)은 notch filter(50 Hz)만 적용한 raw EEG 및 EOG 신호만을 입력으로 사용하였다.

### CNN 아키텍처

제안된 모델은 4개의 Convolution layer, 2개의 Max-pooling layer, 그리고 2개의 Fully-connected layer로 구성된다. 구체적인 설계 특징은 다음과 같다.

- **첫 번째 Layer**: 입력 데이터의 샘플링 레이트가 $100\text{ Hz}$이고 고려하는 최저 주파수가 $0.5\text{ Hz}$이므로, 이를 포착하기 위해 커널 사이즈를 200으로 설정하였다.
- **이후 Layer**: 층이 깊어질수록 커널 사이즈를 조정하여 시간적 정보부터 주파수 도메인의 특징까지 단계적으로 학습하도록 설계하였다.
- **출력**: 최종 Softmax layer를 통해 AASM(American Academy of Sleep Medicine) 기준에 따른 5가지 수면 단계(W, N1, N2, N3, REM)를 분류한다.

### 학습 절차 및 손실 함수

- **손실 함수**: 다중 클래스 분류를 위해 Cross-entropy loss를 사용하였다.
- **최적화**: Adam optimizer를 사용하였으며, 학습률(learning rate)은 $0.00001$, weight decay는 $0.003$, 배치 사이즈는 10으로 설정하여 50 epoch 동안 학습하였다.
- **검증**: 20-fold cross-validation을 통해 성능을 평가하였다.
- **평가 지표**: Accuracy와 함께 우연에 의한 일치 확률을 보정한 $\kappa\text{-value}$를 사용하였다. 수식은 다음과 같다.
$$\kappa = \frac{P_o - P_e}{1 - P_e} = 1 - \frac{1 - P_o}{1 - P_e}$$
여기서 $P_o$는 관측된 정확도(accuracy)이며, $P_e$는 우연히 일치할 확률(probability of chance rate)을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: sleep-edf (expanded) 데이터셋을 사용하였으며, 로드 문제가 있는 데이터를 제외한 145명의 PSG 데이터를 활용하였다.
- **비교 대상**: Raw signal을 사용한 대조군(Control)과 제안된 spectral-temporal 특징을 사용한 실험군(Experimental)을 비교하였다.

### 정량적 결과

실험 결과, 제안된 방법이 대조군 및 기존 연구들보다 우수한 성능을 보였다.

- **전체 정확도(Overall Accuracy)**: 대조군 $85.6\%$ $\rightarrow$ 실험군 $91.1\%$
- **$\kappa\text{-value}$**: 대조군 $0.82$ $\rightarrow$ 실험군 $0.889$
- **단계별 성능**: 특히 N1 단계의 정확도가 대조군 $58.5\%$에서 실험군 $71.5\%$로 크게 향상되었으며, N3 단계에서도 $95.8\%$라는 높은 성능을 기록하였다.

### 기존 연구와의 비교

Table II에 따르면, 제안된 방법은 N1, N3 및 전체 정확도 측면에서 기존의 DeepSleepNet, Phan et al., Yildirim et al. 등의 모델보다 높은 수치를 기록하였다. 다만, REM과 N2 단계에서는 일부 기존 모델보다 상대적으로 낮은 성능을 보였다.

### PSD 분석

Power Spectral Density (PSD) 분석을 통해 각 수면 단계별 주파수 특성을 확인하였다. W에서 N2로 갈수록 Alpha 대역의 전력이 약해지고 Sleep spindle 대역이 강해지며, N3에서는 Delta 전력이 압도적으로 증가하고, REM에서는 Theta와 Alpha 전력이 다시 증가하는 특성이 확인되었다. 이는 제안한 입력 특징 구성이 타당함을 뒷받침한다.

## 🧠 Insights & Discussion

본 논문은 단순한 모델 구조의 변경이 아니라, 수면 과학의 도메인 지식을 입력 데이터 구성에 반영함으로써 성능을 비약적으로 향상시켰다는 점에 강점이 있다. 특히 기존 모델들이 어려워하던 N1 단계의 분류 성능을 크게 끌어올린 것은 spectral-temporal 특징이 유효했음을 시사한다.

그러나 몇 가지 한계점이 존재한다.

1. **클래스 불균형**: N1 단계의 샘플 수가 다른 단계에 비해 너무 적어, 절대적인 성능 향상에도 불구하고 여전히 다른 단계보다 낮은 정확도를 보인다. 저자는 이를 해결하기 위해 향후 데이터 증강(data augmentation)이 필요함을 언급하였다.
2. **Trade-off 발생**: N1의 성능을 높이는 과정에서 N2와 REM 단계의 성능이 상대적으로 낮아지는 트레이드-오프 현상이 관찰되었다.
3. **데이터셋 제한**: sleep-edf 데이터셋만을 사용했으므로, 일반화 성능 검증을 위해 다른 수면 데이터셋에 대한 추가 실험이 필요하다.

결론적으로, 본 연구는 네트워크 전체를 수정하지 않고도 입력 특징의 최적화만으로 성능을 높일 수 있음을 보여주었으며, 이는 향후 다른 수면 단계 분류 프레임워크에도 적용 가능한 유용한 접근 방식이다.

## 📌 TL;DR

본 논문은 수면 단계별 고유 주파수 특성(Delta, Theta, Alpha, Sleep Spindle)을 Bandpass Filter로 추출하여 입력 데이터에 포함시킨 CNN 기반의 자동 수면 단계 분류 프레임워크를 제안한다. 이를 통해 기존 raw signal 기반 방식보다 향상된 $91.1\%$의 정확도를 달성하였으며, 특히 분류가 어려웠던 N1 단계의 성능을 크게 개선하였다. 이 연구는 도메인 지식을 딥러닝 입력 설계에 반영하는 것이 실용적인 성능 향상에 얼마나 중요한지를 입증하며, 향후 수면 진단 보조 도구로서의 활용 가능성을 보여준다.
