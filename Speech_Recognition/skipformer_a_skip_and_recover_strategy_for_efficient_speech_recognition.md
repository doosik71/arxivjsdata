# SKIPFORMER: A SKIP-AND-RECOVER STRATEGY FOR EFFICIENT SPEECH RECOGNITION

Wenjing Zhu, Sining Sun, Changhao Shan, Peng Fan, Qing Yang (2024)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 분야에서 널리 사용되는 Conformer 기반의 Attention 모델이 가진 계산 효율성 문제를 해결하고자 한다. Conformer의 Attention 메커니즘은 입력 시퀀스 길이에 따라 계산 복잡도와 메모리 소비가 제곱 비례(quadratically)하여 증가하는 특성이 있어, 긴 입력 시퀀스를 처리할 때 상당한 계산 비용이 발생한다.

특히 ASR 모델에서 입력 시퀀스는 출력 시퀀스보다 훨씬 길며, CTC(Connectionist Temporal Classification)나 RNN-T 모델에서는 입력과 출력의 정렬을 위해 대량의 blank symbol을 도입한다. 이러한 blank symbol이 포함된 프레임들은 최종 인식 성능에는 기여도가 낮으면서도 불필요한 연산을 유발하므로, 이를 효율적으로 처리하여 추론 속도를 높이고 계산 비용을 줄이는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"프레임이 가진 정보의 중요도에 따라 서로 다른 복잡도의 모델을 적용한다"**는 것이다. 즉, 정보량이 적은 프레임은 단순한 처리만 거치게 하고, 중요한 정보를 가진 프레임은 더 복잡한 모델을 통해 심층적으로 분석하는 전략이다.

이를 위해 제안된 **"Skip-and-Recover"** 전략은 중간 단계의 CTC 출력을 기준으로 프레임을 세 가지 그룹(Crucial, Skipping/Trivial, Ignoring)으로 동적으로 분류한다. 중요한 프레임(Crucial)만 후속 인코더 블록에서 처리하고, 덜 중요한 프레임(Trivial)은 해당 블록을 건너뛰어(Skip) 나중에 결합하며, 무의미한 프레임(Ignoring)은 완전히 제거함으로써 시퀀스 길이를 획기적으로 줄인다.

## 📎 Related Works

기존의 계산 효율성 개선 연구들은 주로 고정된 하향 샘플링(Downsampling) 계수를 사용하는 방식이었다.

- **Vanilla Conformer**: Convolution 레이어를 통해 입력 길이를 4배 감소시킨다.
- **Efficient Conformer**: 점진적 하향 샘플링을 통해 길이를 8배까지 줄였다.
- **Squeezeformer 및 Uconv-Conformer**: Temporal U-Net 구조나 심층 하향 샘플링 후 상향 샘플링(Upsampling)을 통해 해상도를 회복하는 방식을 사용했다.

이러한 방식들은 프레임을 균일하게(uniformly) 줄이는 한계가 있다. 일부 연구에서 CTC 가이드를 통해 중요도 기반 샘플링(Importance sampling)을 시도하여 blank 프레임을 제거했으나, AED(Attention-based Encoder-Decoder) 모델에 이를 단순 적용했을 때 인식 성능이 크게 저하되는 문제가 있었다. Skipformer는 단순히 프레임을 버리는 것이 아니라, 일부는 유지하고 일부는 건너뛰는 "Skip-and-Recover" 방식을 통해 성능 저하 없이 효율성을 달성하며 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

Skipformer는 Joint CTC and AED 구조를 채택하며, 인코더를 두 개의 서브 인코더 $E_1$과 $E_2$로 분리한다. 전체적인 데이터 흐름은 다음과 같다.

1. **초기 처리**: 입력 신호 $x$는 `Conv2dSubsampling`을 통해 $x_s$로 변환된다.
2. **1차 인코딩**: $x_s$는 $M$개의 블록으로 구성된 서브 인코더 $E_1$을 통과하여 $h_1$을 생성한다.
3. **프레임 분류 (Split)**: $h_1$에 대한 중간 CTC 출력 확률을 기준으로 프레임을 세 그룹으로 나눈다.
   - **Crucial group ($h_1^c$)**: 의미적 정보가 풍부한 프레임들.
   - **Trivial group ($h_1^t$)**: 반복 기호를 구분하는 경계 역할을 하는 프레임들.
   - **Ignoring group ($h_1^i$)**: 완전히 불필요하여 제거될 프레임들.
4. **2차 인코딩 (Skip)**: 오직 **Crucial group ($h_1^c$)**만이 $N$개의 블록으로 구성된 서브 인코더 $E_2$를 통과하여 $h_2^c$가 된다. Trivial group은 $E_2$의 복잡한 연산을 건너뛴다.
5. **복구 (Recover)**: $E_2$를 통과한 $h_2^c$와 건너뛴 $h_1^t$를 원래의 시간 순서대로 다시 결합하여 최종 인코더 출력 $h_2$를 생성한다.
6. **디코딩**: 최종 출력 $h_2$는 Attention Decoder로 전달되어 텍스트로 변환된다.

### 주요 방정식

전체 과정은 다음과 같은 수식으로 표현된다.

$$x_s = \text{Conv2dSubsampling}(x)$$
$$h_1 = E_1(x_s)$$
$$h_1^c, h_1^t, h_1^i = \text{Split}(h_1 | \text{CTC}(h_1))$$
$$h_2^c = E_2(h_1^c)$$
$$h_2 = \text{Recover}(h_2^c, h_1^t)$$

### 학습 목표 및 손실 함수

중간 CTC 출력은 프레임 분류의 기준이 됨과 동시에 모델의 성능을 높이는 정규화 역할을 한다. 전체 손실 함수 $L$은 다음과 같이 정의된다.

$$L = \alpha(\lambda_1 L_{\text{inter}}^{\text{CTC}} + \lambda_2 L_{\text{final}}^{\text{CTC}}) + (1-\alpha)(\lambda_1 L_{\text{inter}}^{\text{AED}} + \lambda_2 L_{\text{final}}^{\text{AED}})$$

여기서 $L_{\text{inter}}$는 서브 인코더 $E_1$ 이후의 중간 손실이며, $L_{\text{final}}$은 최종 출력 이후의 손실이다. $\alpha, \lambda_1, \lambda_2$는 하이퍼파라미터이다.

### 데이터 분할 전략 (Split Mode)

논문은 blank 프레임을 어떻게 분류할지에 대해 5가지 모드를 제안하며, 특히 **Mode 2**가 가장 효율적임을 밝혔다.

- **Mode 2**: Non-blank 프레임은 $h_1^c$로, 각 non-blank 프레임의 오른쪽 인접 blank 프레임 하나만 $h_1^t$로 분류하고, 나머지 모든 blank 프레임은 $h_1^i$로 분류하여 제거한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Aishell-1 (중국어), Librispeech (영어)
- **지표**: Character Error Rate (CER), Word Error Rate (WER), Inverse Real Time Factor (Inv RTF)
- **모델 구성**: 총 12개 인코더 블록 (Aishell-1: $M=5, N=7$ / Librispeech: $M=6, N=6$), 6개 디코더 블록.
- **기준선**: Vanilla Conformer, Efficient Conformer, Squeezeformer.

### 주요 결과

1. **인식 정확도**: Aishell-1 데이터셋에서 Skipformer는 4.23%의 CER을 달성하여 Vanilla 및 Efficient Conformer보다 우수한 성능을 보였다. Librispeech에서도 Squeezeformer 대비 대등하거나 더 나은 WER을 기록했다.
2. **추론 속도 (Inv RTF)**:
   - **Aishell-1**: 평균 시퀀스 길이를 약 31배 감소시켰으며, CPU 및 GPU 환경 모두에서 Inv RTF가 크게 향상되었다.
   - **Librispeech**: 시퀀스 길이를 약 22배 감소시켰으며, Squeezeformer 대비 CPU에서 47%, GPU에서 56% 더 빠른 추론 속도를 보였다.
3. **Ablation Study**:
   - $M$과 $N$의 비율에 따라 $N$(후속 블록)이 많을수록 계산 비용이 줄어들지만, $M=5, N=7$ 설정에서 최적의 CER과 효율성 균형을 찾았다.
   - Split Mode 실험 결과, Mode 2가 정확도와 효율성 사이의 최적의 트레이드오프를 제공함을 확인했다.

## 🧠 Insights & Discussion

### 강점

본 연구는 무조건적인 다운샘플링이 아니라, CTC의 확률값을 이용해 프레임의 중요도를 동적으로 판단했다는 점에서 매우 영리한 접근 방식을 취하고 있다. 특히, blank 프레임을 모두 제거하면 성능이 떨어지는 문제를 'Trivial' 그룹으로 분류해 보존함으로써 해결한 점이 인상적이다. 이는 모델이 연산량은 줄이면서도 시퀀스의 시간적 구조(Temporal structure)를 유지할 수 있게 한다.

### 한계 및 논의사항

- **임계값 의존성**: 프레임을 분류할 때 사용되는 blank 확률 임계값 $\beta$ (기본값 0.99)에 대한 민감도 분석이 구체적으로 제시되지 않았다. 이 값에 따라 연산량과 성능이 크게 변할 가능성이 있다.
- **학습 복잡도**: 중간 CTC 손실을 추가하고 복잡한 분할 전략을 사용하므로, 추론 속도는 빨라지지만 학습 단계에서의 최적화 난이도나 학습 시간 증가 여부에 대한 언급이 부족하다.
- **범용성**: 특정 데이터셋(Aishell-1, Librispeech)에서 성능이 입증되었으나, 소음이 심한 환경이나 매우 짧은 발화 등 다양한 도메인에서도 동일한 효율성이 유지될지는 추가 검증이 필요하다.

## 📌 TL;DR

Skipformer는 Conformer 인코더를 두 단계로 나누고, 중간 CTC 출력을 이용해 프레임을 **Crucial(심층 처리), Trivial(건너뛰기), Ignoring(제거)**으로 분류하는 "Skip-and-Recover" 전략을 제안한다. 이를 통해 Aishell-1에서 31배, Librispeech에서 22배까지 시퀀스 길이를 줄이면서도 기존 SOTA 모델보다 더 높은 인식 정확도와 획기적으로 빠른 추론 속도를 달성했다. 이 연구는 실시간 음성 인식 시스템의 하드웨어 자원 제약을 극복하는 데 매우 중요한 기여를 할 것으로 보인다.
