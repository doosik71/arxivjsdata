# Friends Across Time: Multi-Scale Action Segmentation Transformer for Surgical Phase Recognition

Bokai Zhang, Jiayuan Meng, Bin Cheng, Dean Biskup, Svetlana Petculescu, Angela Chapman (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 수술 비디오에서 각 수술 단계(Surgical Phase)의 시작과 종료 시간을 자동으로 감지하는 '수술 단계 인식(Surgical Phase Recognition)'이다. 수술 단계 인식은 현대 수술실의 핵심 기술이며, 온라인 환경에서는 집도의를 실시간으로 보조하고, 오프라인 환경에서는 수술 비디오 컬렉션을 효율적으로 분류하여 수술 숙련도 평가 및 수술 기법의 표준화를 가능하게 한다.

기존의 연구들은 공간적 정보와 시간적 정보를 결합하여 이 문제에 접근해 왔으나, 수술 단계는 매우 짧은 동작(fast action)부터 매우 긴 단계(slow action)까지 그 시간적 규모(temporal scale)가 매우 다양하다는 특성이 있다. 따라서 단일한 시간적 스케일만으로는 이러한 다양한 길이의 수술 단계를 정밀하게 포착하는 데 한계가 있으며, 이는 과분할(over-segmentation)이나 순서 오류(out-of-order predictions)와 같은 문제로 이어진다. 본 논문의 목표는 다양한 시간적 스케일을 동시에 모델링할 수 있는 트랜스포머 구조를 설계하여, 온라인과 오프라인 환경 모두에서 수술 단계 인식의 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **다중 스케일 시간적 어텐션(Multi-scale Temporal Attention)** 메커니즘을 통해 프레임 수준의 미세한 변화와 세그먼트 수준의 거시적 흐름을 동시에 캡처하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **MS-AST (Multi-Scale Action Segmentation Transformer) 제안**: ASFormer 구조를 확장하여 다중 스케일 시간적 셀프 어텐션(Multi-scale Temporal Self-Attention)과 크로스 어텐션(Multi-Scale Temporal Cross-Attention)을 도입함으로써, 서로 다른 시간적 범위의 관계를 동시에 학습할 수 있도록 설계하였다.
2. **MS-ASCT (Multi-Scale Action Segmentation Causal Transformer) 제안**: MS-AST를 기반으로 하되, 미래 정보의 유출을 방지하는 인과적(Causal) 설계를 적용하여 실시간 수술실 환경에서 사용할 수 있는 온라인 인식 모델을 구현하였다.
3. **SOTA 성능 달성**: Cholec80 데이터셋에서 온라인 및 오프라인 수술 단계 인식 모두에서 새로운 State-of-the-art(SOTA) 결과를 달성하였다.
4. **범용성 검증**: 의료 데이터뿐만 아니라 일반 액션 세그멘테이션 데이터셋인 50Salads와 GTEA에서도 SOTA 성능을 기록함으로써, 제안 방법론의 견고함과 범용성을 입증하였다.

## 📎 Related Works

수술 단계 인식 및 액션 세그멘테이션 분야에서는 초기 이미지 분류 네트워크 기반의 방법론에서 시작하여, LSTM과 같은 순환 신경망(RNN)을 결합한 방식이 사용되었다. 이후 MS-TCN(Multi-Stage Temporal Convolutional Networks)과 같은 다단계 시간적 합성곱 신경망이 등장하며 LSTM보다 우수한 성능을 보였다.

최근에는 Vision Transformer의 발전으로 temporal modeling에 트랜스포머를 적용하는 추세이며, 특히 ASFormer는 인코더-디코더 구조와 확장 합성곱(dilated convolution), 슬라이딩 윈도우 어텐션을 사용하여 국소적 특징에서 전역적 정보로 수용 영역(receptive field)을 확장하는 방식을 취했다. 하지만 ASFormer는 단일 스케일의 어텐션을 사용하므로, 다양한 길이의 수술 단계를 동시에 효과적으로 처리하는 데 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 여러 개의 커널 크기와 윈도우 크기를 사용하는 다중 스케일 구조를 도입하여 차별점을 두었다.

## 🛠️ Methodology

### 전체 파이프라인

전체 시스템은 크게 두 단계로 구성된다. 먼저 ResNet50 또는 EfficientNetV2-M과 같은 이미지 분류 네트워크를 사용하여 각 비디오 프레임에서 공간적 특징(spatial feature)을 추출한다. 이후 추출된 프레임 수준의 특징 시퀀스를 MS-AST 또는 MS-ASCT 네트워크에 입력하여 시간적 모델링을 수행하고 최종적으로 수술 단계를 인식한다.

### MS-AST 아키텍처

MS-AST는 인코더-디코더 구조를 가진다. 인코더는 각 프레임의 초기 액션 확률을 예측하며, 디코더는 이 예측값을 점진적으로 정교화(refinement)한다.

1. **Multi-scale Temporal Self-Attention (Encoder)**:
   - 세 가지 서로 다른 시간적 스케일을 사용한다.
   - 확장 합성곱(Dilated Convolution)의 커널 크기를 $3, 5, 17$로 설정하여 각기 다른 스케일의 특징을 추출한다.
   - 슬라이딩 윈도우 어텐션의 윈도우 크기 또한 커널 크기에 따라 다르게 설정된다. 커널 크기가 3일 때는 1에서 512까지, 5일 때는 4에서 1024까지, 17일 때는 16에서 4096까지 층마다 두 배씩 증가한다.
   - 각 스케일에서 계산된 어텐션 결과는 학습 가능한 가중치 $w$와 감쇠 계수 $\alpha$를 통해 결합된다.

$$ \text{out}_i = \alpha \times w_{1,i} \times \text{Attention}_{1,i}(\text{out}_i) + \alpha \times w_{2,i} \times \text{Attention}_{2,i}(\text{out}_i) + \alpha \times w_{3,i} \times \text{Attention}_{3,i}(\text{out}_i) + \text{out}_i $$

여기서 $\text{out}_i$는 $i$번째 블록의 출력이며, $\alpha$는 첫 번째 디코더에서는 1이며 이후 디코더 층으로 갈수록 지수적으로 감소한다.

1. **Multi-scale Temporal Cross-Attention (Decoder)**:
   - 인코더의 출력과 이전 디코더 층의 출력을 Query($Q$)와 Key($K$)로 사용하고, 이전 층의 출력을 Value($V$)로 사용하여 동일한 다중 스케일 메커니즘을 적용한다.

### MS-ASCT (Online Version)

실시간 인식을 위해 MS-AST를 다음과 같이 수정하여 인과성(Causality)을 부여하였다.

- **Causal Dilated Convolution**: 현재 시점의 예측이 미래 프레임의 정보에 의존하지 않도록 인과적 합성곱을 사용한다.
- **Layer Normalization 제거**: 미래 정보가 현재로 누설되는 것을 방지하기 위해 레이어 정규화를 제거한다.
- **Causal Sliding Window Attention**: 어텐션 연산 시 오직 과거와 현재의 정보만을 참조하도록 윈도우 범위를 제한한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80 (수술), 50Salads 및 GTEA (일반 액션).
- **지표**: 프레임 수준의 Accuracy, Precision, Recall, Jaccard score 및 세그먼트 수준의 Edit distance, F1-score ($\text{F1}_{\text{AVG}}$)를 사용하였다.
- **비교 대상**: PhaseLSTM, EndoLSTM, SV-RCNet, TeCNO, ST-CT, ASFormer 등 기존 SOTA 모델들.

### 주요 결과

1. **온라인 수술 단계 인식 (Cholec80)**:
   - EffNetV2 MS-ASCT 모델이 **95.26%의 정확도**를 기록하며 기존 SOTA인 ST-CT 및 C-ECT를 상회하였다.
   - 특히 세그먼트 기반 지표인 $\text{F1}_{\text{AVG}}$에서 기존 모델 대비 약 8% 이상의 향상을 보였으며, 이는 과분할 에러가 크게 감소했음을 의미한다.

2. **오프라인 수술 단계 인식 (Cholec80)**:
   - EffNetV2 MS-AST 모델이 **96.15%의 정확도**를 달성하여 ResNet ASFormer 등 기존 모델보다 우수한 성능을 보였다.

3. **비의료 데이터셋 검증**:
   - 50Salads 데이터셋에서 90.5% Accuracy, GTEA 데이터셋에서 82.3% Accuracy를 기록하며 일반 액션 세그멘테이션 영역에서도 SOTA 수준의 성능을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 단일 스케일의 어텐션만 사용하던 기존 트랜스포머 모델에 다중 스케일 구조를 도입함으로써, 수술 비디오 특유의 다양한 시간적 길이를 효과적으로 처리할 수 있음을 보여주었다. 시각화 결과(Figure 4)를 통해 MS-ASCT가 기존 Causal ASFormer보다 과분할 에러와 예측 순서 오류를 현저히 줄였음을 확인할 수 있다. 이는 서로 다른 크기의 수용 영역을 동시에 활용하는 것이 수술 단계의 경계선을 더 명확하게 구분 짓는 데 기여했음을 시사한다.

### 한계 및 비판적 논의

Confusion Matrix 분석 결과, P3(Clipping and cutting)를 P4(Gallbladder dissection)로 예측하거나, P6와 P7 사이에서 혼동이 발생하는 경우가 관찰되었다. 저자들은 P3와 P7과 같은 짧은 단계의 학습 데이터가 부족한 것이 원인일 수 있다고 언급한다. 이는 모델의 구조적 개선뿐만 아니라, 데이터 불균형 문제를 해결하기 위한 데이터 증강(Augmentation)이나 손실 함수(Loss function)의 개선이 추가적으로 필요함을 시사한다. 또한, 실시간 환경에서의 실제 추론 속도(Inference Latency)에 대한 정량적 분석이 부족하여 실제 수술실 적용 가능성을 완전히 판단하기에는 정보가 제한적이다.

## 📌 TL;DR

본 논문은 수술 비디오의 다양한 시간적 규모를 캡처하기 위해 다중 스케일 어텐션을 도입한 **MS-AST**(오프라인용)와 **MS-ASCT**(온라인용)를 제안하였다. 제안된 방법은 Cholec80 데이터셋에서 각각 96.15%와 95.26%의 정확도를 기록하며 SOTA 성능을 달성하였으며, 일반 액션 세그멘테이션 데이터셋에서도 우수한 성능을 보였다. 이 연구는 다양한 길이의 단계가 혼재된 복잡한 시퀀스 분석 문제에서 다중 스케일 모델링의 중요성을 입증하였으며, 향후 실시간 수술 보조 시스템 구축에 중요한 기반이 될 것으로 기대된다.
