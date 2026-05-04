# ARB-LLM: Alternating Refined Binarizations for Large Language Models

Zhiteng Li, Xianglong Yan, Tianao Zhang, Haotong Qin, Dong Xie, Jiang Tian, Zhongchao Shi, Linghe Kong, Yulun Zhang, Xiaokang Yang (2024)

## 🧩 Problem to Solve

거대 언어 모델(Large Language Models, LLMs)은 자연어 처리 분야에서 혁신적인 성능을 보여주었으나, 수십억 개의 파라미터로 인한 막대한 메모리 요구량과 계산 비용이 실제 배포의 큰 걸림돌이 된다. 이를 해결하기 위해 가중치를 1비트로 압축하는 Binarization 기술이 주목받고 있으며, 이는 메모리 사용량을 획기적으로 줄일 수 있는 효율적인 방법이다.

그러나 기존의 Binarization 방법들은 다음과 같은 두 가지 핵심적인 문제점을 가지고 있다. 첫째, 이진화된 가중치와 원래의 Full-precision 가중치 사이의 분포 차이(Distribution gap)를 좁히는 데 어려움이 있어 양자화 오차가 크게 발생한다. 둘째, LLM 가중치 분포에서 나타나는 열 단위의 편차(Column deviation)를 간과하여 모델의 성능 저하를 초래한다. 본 논문의 목표는 이러한 분포 불일치와 열 편차 문제를 해결하여, FP16 모델에 근접하거나 이를 능가하는 성능을 가진 1-bit Post-Training Quantization(PTQ) 기법인 ARB-LLM을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이진화 파라미터를 점진적으로 업데이트하여 Full-precision 가중치의 분포를 정밀하게 추적하는 **Alternating Refined Binarization (ARB)** 프레임워크를 설계한 것이다.

중심적인 설계 직관은 다음과 같다.
1. **점진적 정렬**: 한 번의 이진화로 끝내는 것이 아니라, 평균($\mu$), 스케일링 인자($\alpha$), 이진 행렬($B$)을 교대로 업데이트하여 양자화 오차를 최소화한다.
2. **데이터 기반 정교화**: 단순 가중치 오차뿐만 아니라 실제 Calibration 데이터를 활용하여 가중치가 입력 데이터와 상호작용하는 실제 영향을 반영한다(ARB-X).
3. **열 단위 스케일링**: 행 단위 스케일링만으로는 부족한 LLM의 특성을 반영하여 행과 열 모두에 스케일링 인자를 도입한다(ARB-RC).
4. **효율적인 비트맵 관리**: Salient weight(중요 가중치)와 Non-salient weight를 구분하고, 이를 다시 그룹화하는 Column-Group Bitmap(CGB) 전략을 통해 비트맵 활용도를 극대화한다.

## 📎 Related Works

### 기존 연구 및 한계
- **Network Binarization**: XNOR-Net과 같은 초기 연구들은 $\text{sign}$ 함수와 단일 스케일링 인자 $\alpha$를 사용하여 가중치와 활성화 함수를 이진화했다. 하지만 이는 주로 학습 단계에서 이루어지며, LLM과 같은 거대 모델에 적용하기에는 계산 자원이 너무 많이 소요된다.
- **LLM Quantization (QAT vs PTQ)**: Quantization-Aware Training(QAT)은 성능이 좋지만 재학습 비용이 크다. 반면 Post-Training Quantization(PTQ)은 빠르게 적용 가능하지만, 1-bit 수준의 극단적인 압축에서는 정보 손실이 심각하다.
- **Binary PTQ (PB-LLM, BiLLM)**: 최근 연구들은 일부 중요 가중치(Salient weights)만 높은 비트로 유지하고 나머지를 이진화하는 하이브리드 방식을 사용했다. 하지만 이들 역시 이진화 과정 자체의 정밀한 최적화(Refinement)보다는 어떤 가중치를 살릴 것인가에 집중했으며, 이로 인해 여전히 Full-precision 가중치와의 분포 차이가 존재한다.

### 차별점
ARB-LLM은 단순한 가중치 선택을 넘어, **교대 업데이트(Alternating update)**라는 최적화 절차를 통해 이진화된 가중치가 원래의 분포를 가장 잘 모사하도록 강제한다는 점에서 기존 PTQ 방식과 차별화된다.

## 🛠️ Methodology

### 1. Alternating Refined Binarization (ARB)
기본적인 이진화 목적 함수는 다음과 같다.
$$\arg \min_{\alpha, B} \|f_W - \alpha B\|_F^2, \quad \text{where } f_W = W - \mu$$
여기서 $\mu$는 행 평균, $\alpha$는 행별 스케일링 인자, $B \in \{+1, -1\}^{n \times m}$이다.

ARB는 다음과 같은 단계로 파라미터를 반복적으로 업데이트한다.
1. **평균 정교화 ($\mu$ update)**: 이진화 후의 잔차 행렬 $R = W - \hat{W}$의 평균 $\delta_\mu$를 계산하여 기존 $\mu$에 더해준다.
   $$\mu_{\text{refine}} = \mu + \delta_\mu, \quad \delta_\mu = \frac{1}{m} \sum_{j=1}^m R_{.j}$$
2. **스케일링 인자 정교화 ($\alpha$ update)**: $\partial L_1 / \partial \alpha = 0$이 되는 지점을 찾아 업데이트한다.
3. **이진 행렬 정교화 ($B$ update)**: 최신 $\mu_{\text{refine}}$을 기준으로 $\text{sign}$ 함수를 통해 $B$를 갱신한다.
   $$B_{\text{refine}} = \text{sign}(W - \mu_{\text{refine}})$$

이 과정을 $\tau$회 반복함으로써 양자화 오차 $L_1$을 점진적으로 줄인다.

### 2. ARB-X (Calibration Data 활용)
가중치 자체의 오차($L_1$)보다 입력 데이터 $X$가 포함된 출력 오차 $L_2 = \|WX - \hat{W}X\|_F^2$를 최소화하는 것이 더 중요하다. 하지만 이를 직접 계산하면 연산량이 너무 많으므로, 다음과 같이 수식을 재구성(Reformulation)하여 속도를 높였다.
$$L_2 = \text{Tr}(RSR^\top), \quad \text{where } S = \sum X_b X_b^\top$$
여기서 $S$는 미리 계산된 행렬이므로, 반복적인 업데이트 과정에서 계산 복잡도를 획기적으로 줄여 약 389배의 속도 향상을 달성했다.

### 3. ARB-RC (Row-Column Scaling)
LLM 가중치의 열 단위 편차를 해결하기 위해 평균 $\mu$를 제거하는 대신, 행 스케일링 $\alpha_r$과 열 스케일링 $\alpha_c$를 동시에 도입한다.
$$\hat{W} = \alpha_r \alpha_c B$$
$\alpha_r$과 $\alpha_c$를 교대로 업데이트하며 최적의 값을 찾는다.
$$\alpha_r = \frac{\text{diag}(W(\alpha_c B)^\top)}{\text{diag}((\alpha_c B)(\alpha_c B)^\top)}, \quad \alpha_c = \frac{\text{diag}(W^\top(\alpha_r B))}{\text{diag}((\alpha_r B)^\top(\alpha_r B))}$$

### 4. Column-Group Bitmap (CGB)
중요 가중치를 식별하는 Hessian 기반의 Salient Column Bitmap과 가중치 크기 기반의 Group Bitmap을 결합하는 전략이다. 기존 BiLLM은 Non-salient weight만 그룹화했으나, CGB는 **Salient weight까지도 그룹화**하여 비트맵 저장 공간을 효율적으로 사용하고 양자화 정확도를 높였다.

## 📊 Results

### 실험 설정
- **모델**: OPT family, LLaMA (1, 2, 3), Vicuna.
- **데이터셋**: WikiText2, PTB, C4 (Perplexity 측정) 및 7개의 Zero-shot QA 데이터셋 (Accuracy 측정).
- **비교 대상**: RTN, GPTQ, PB-LLM, BiLLM.

### 주요 결과
1. **Perplexity (PPL)**:
   - OPT-66B 모델에서 $\text{ARB-LLM}_{RC}$는 3-bit GPTQ보다 낮은 PPL을 기록했다.
   - LLaMA-70B 등 대부분의 모델에서 SOTA인 BiLLM보다 월등히 낮은 PPL을 보여주며, 최대 68.7%의 PPL 감소를 달성했다.
2. **Zero-shot QA Accuracy**:
   - $\text{ARB-LLM}_{RC}$는 **동일 크기의 FP16 모델보다 더 높은 평균 정확도**를 기록했다. 이는 Binary PTQ 방법론으로서는 최초의 성과이다.
3. **효율성**:
   - **메모리**: $\text{ARB-LLM}_{RC}$는 $\mu$를 제거하고 $\alpha_c$를 도입함으로써 BiLLM보다 더 적은 메모리를 사용하면서도 성능은 더 뛰어나다.
   - **시간**: 반복적인 업데이트로 인해 BiLLM보다 시간이 더 소요되지만(LLaMA-7B 기준 약 21분 추가), 이는 실용적인 수준의 오버헤드이다.

## 🧠 Insights & Discussion

### 강점
본 논문은 단순한 하이퍼파라미터 튜닝이 아니라, 이진화 과정에서 발생하는 **분포 시프트(Distribution shift)**라는 근본적인 원인을 분석하고 이를 수학적인 반복 최적화(Alternating optimization)로 해결했다는 점에서 학술적 가치가 높다. 특히 Row-Column scaling의 도입이 LLM의 특이한 가중치 분포를 잡는 데 매우 효과적이었음을 입증했다.

### 한계 및 논의
- **Calibration 데이터 의존성**: ARB-X의 경우 Calibration 데이터의 크기가 성능에 영향을 미친다. 비록 128개 샘플로 충분하다고 주장하지만, 데이터셋의 성격에 따라 성능 변동이 있을 수 있다.
- **CGB의 복잡성**: CGB 전략은 성능을 높이지만 비트맵 저장 공간과 최적 분할 지점을 찾는 계산 시간이 추가된다.
- **해석**: Binary 모델이 FP16 모델을 능가하는 결과는 매우 놀랍지만, 이는 Binarization 과정에서의 정규화 효과(Regularization effect)가 작용했을 가능성이 있으며, 이에 대한 추가적인 분석이 필요해 보인다.

## 📌 TL;DR

본 논문은 LLM의 1-bit 압축 시 발생하는 분포 불일치와 열 편차 문제를 해결하기 위해, 파라미터를 교대로 업데이트하는 **ARB-LLM** 프레임워크를 제안한다. 특히 행-열 동시 스케일링($\text{ARB-RC}$)과 정교한 비트맵 전략($\text{CGB}$)을 결합하여, **최초로 동일 크기의 FP16 모델 성능을 능가하는 1-bit PTQ 모델을 구현**했다. 이는 거대 모델의 초경량 배포 가능성을 크게 확장시킨 연구이다.