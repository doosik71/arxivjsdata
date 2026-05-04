# Training-free Token Reduction for Vision Mamba

Qiankun Ma, Ziyao Zhang, Chi Su, Jie Chen, Zhen Song, Hairong Zheng, Wen Gao (2025)

## 🧩 Problem to Solve

본 논문은 Vision Mamba(ViM) 모델의 추론 효율성을 높이기 위한 토큰 감소(Token Reduction) 기법을 다룬다. Vision Mamba는 선형 계산 복잡도를 통해 긴 범위의 의존성을 효율적으로 포착함으로써 Vision Transformer(ViT)의 강력한 대안으로 부상하였다. 하지만 ViT에서 효과적으로 사용되던 토큰 감소 기술들을 Vision Mamba에 직접 적용했을 때 심각한 성능 저하가 발생하는 문제가 발견되었다.

이러한 성능 저하의 원인은 크게 두 가지이다. 첫째, Mamba는 시퀀스 모델이므로 토큰의 순서(order)가 성능에 결정적인 영향을 미치지만, 기존 ViT 기반의 기법들은 토큰의 순서를 무시하는 경향이 있다. 둘째, 대부분의 ViT 토큰 감소 기법은 Attention 메커니즘의 가중치를 통해 토큰의 중요도를 측정하는데, Mamba는 Attention 메커니즘이 없는 구조이므로 이러한 지표를 직접 사용할 수 없다. 따라서 본 논문의 목표는 Mamba의 구조적 특성을 반영하여 추가 학습 없이(training-free) 적용 가능한 효율적인 토큰 감소 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba의 내부 파라미터인 timescale parameter $\Delta$가 토큰의 중요도를 측정하는 "Attention score"와 유사한 역할을 수행한다는 점을 발견하고, 이를 활용한 **MTR(Mamba Token Reduction)** 프레임워크를 제안한 것이다.

MTR의 중심 아이디어는 $\Delta$를 통해 토큰의 중요도를 평가하고, 이를 기반으로 토큰을 세 가지 그룹('Keep', 'Target', 'Source')으로 비대칭적으로 분류한 뒤, 중요도가 가장 낮은 'Source' 그룹의 토큰들을 유사도 기반으로 'Target' 그룹에 병합하는 것이다. 이 과정은 추가적인 학습이나 튜닝 파라미터 없이 플러그 앤 플레이(plug-and-play) 방식으로 다양한 Mamba 모델에 통합될 수 있다.

## 📎 Related Works

### Vision Mamba 및 SSM
Mamba는 상태 공간 모델(State Space Model, SSM)을 확장하여 선형 복잡도로 긴 시퀀스를 처리한다. ViM, VMamba, PlainMamba 등 다양한 시각적 Mamba 모델들이 제안되었으나, 대부분은 모델 구조나 스캔 메커니즘(scanning mechanism)의 최적화에 집중했을 뿐 추론 효율성을 위한 토큰 감소 연구는 부족한 상태였다.

### 기존 토큰 감소 기법 및 한계
ViT에서는 EViT([CLS] 토큰 활용), ToMe(유사도 기반 병합), PuMer(텍스트 기반 가지치기 및 병합) 등 다양한 기법이 사용되었다. 그러나 앞서 언급했듯이, 이러한 기법들은 Attention 맵에 의존하거나 토큰의 순서 변경에 둔감한 Transformer 구조를 전제로 설계되었다. Mamba에서도 HSA와 같은 토큰 가지치기 연구가 있었으나, 이는 재학습(retraining)이 필요하다는 한계가 있었다. MTR은 재학습 없이 Mamba의 내부 구조를 활용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### Preliminary: Selective State Space Model
기존의 연속적인 SSM 시스템은 다음과 같이 정의된다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

이를 딥러닝에 적용하기 위해 timescale parameter $\Delta$를 사용하여 이산화(discretization)하면 다음과 같이 변환된다.
$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

Mamba는 여기서 $\Delta, B, C$를 입력 데이터 $x$에 의존적인 파라미터 $\Delta_t, B_t, C_t$로 만들어 선택적 상태 공간 모델(Selective SSM)을 구축한다. 최종적으로 입력 $x_t$는 다음과 같은 형태로 처리된다.
$$h_t = \bar{A}_t \odot h_{t-1} + B_t(\Delta_t \odot x_t)$$
$$y_t = C_t h_t + D \odot x_t$$

### Mamba 구조 기반 토큰 중요도 평가
저자들은 $\Delta_t$가 현재 입력 토큰 $x_t$의 가중치를 조절하는 입력 게이트(input gate) 역할을 한다는 점에 주목하였다. $\Delta_t$ 값이 클수록 현재 입력에 더 집중하고, 작을수록 과거의 기억에 더 의존한다. 따라서 $\Delta_t$를 토큰 중요도 지표로 활용할 수 있다.

각 레이어 $l$에서 스캔 헤드 $S$에 대해 $\Delta$를 합산하고, 특징 차원 $D$에 대해 평균을 내어 최종 중요도 점수 $s^l$을 계산한다.
$$s^l = \frac{1}{D} \sum_{d=1}^{D} \Delta^l_d$$

### MTR 프레임워크 및 압축 절차
MTR은 다음과 같은 단계로 토큰을 감소시킨다.

1.  **중요도 기반 정렬**: 계산된 $s^l$을 기준으로 토큰 시퀀스를 내림차순으로 정렬한다.
2.  **비대칭 그룹화**: 정렬된 토큰들을 중요도에 따라 세 그룹으로 나눈다.
    -   **Keep**: 가장 중요한 토큰들. 그대로 유지된다.
    -   **Target**: 중간 중요도 토큰들. 'Source' 토큰들이 병합될 대상이다.
    -   **Source**: 가장 중요도가 낮은 토큰들. 제거되거나 병합될 대상이다.
3.  **Bipartite Soft Matching 병합**: 'Source' 그룹의 각 토큰 $S_a$에 대해 'Target' 그룹에서 가장 유사한 토큰 $T_b$를 찾아 두 벡터의 평균값으로 병합한다.
4.  **재정렬(Reorder)**: 병합 후 남은 토큰들을 원래의 시퀀스 인덱스 순서대로 다시 배치한다. 이는 Mamba의 순서 민감성 문제를 해결하기 위함이다.

## 📊 Results

### 실험 설정
-   **데이터셋 및 작업**: ImageNet-1K 분류 작업 (Top-1 Accuracy 측정).
-   **백본 모델**: ViM-S, ViM-B, VideoMamba-S, VideoMamba-B.
-   **비교 대상**: EViT, PuMer (ViT 기반), UTR, HSA (Mamba 기반).
-   **지표**: FLOPs 감소율 대비 정확도 유지율.

### 주요 결과
-   **성능 우위**: MTR은 모든 백본 모델과 FLOPs 감소율(20%, 30%, 40%)에서 베이스라인보다 우수한 성능을 보였다. 특히 ViM-B 모델에서 40%의 FLOPs를 줄였을 때, 정확도 하락은 단 1.6%에 불과했다. 반면 UTR과 HSA는 각각 3.9%, 4.2% 하락하였다.
-   **VideoMamba 적용**: 비디오 데이터셋에서도 MTR은 타 방법론 대비 높은 정확도를 유지하며 일반화 능력을 입증하였다.
-   **Ablation Study**:
    -   **지표 검증**: $\Delta_t$가 $B_t, C_t, [CLS]$ 토큰 유사도보다 훨씬 효과적인 중요도 지표임을 확인하였다. 특히 $[CLS]$ 기반 방법은 Mamba의 순차적 특성 때문에 인접 토큰 간 유사도가 높게 나타나 성능이 낮게 측정되었다.
    -   **연산 전략**: 단순 제거(Pruning)보다 병합(Merging) 전략이 정보 손실을 최소화하여 성능이 더 좋음을 확인하였다.
-   **정성적 분석**: 시각화 결과, MTR이 보존하는 'Keep' 그룹의 토큰들이 CAM(Class Activation Map)의 고응답 영역(객체의 핵심 부분)과 일치함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 Mamba 모델에서 무엇이 "중요한 토큰"인지를 정의하기 위해 모델 내부의 $\Delta$ 파라미터를 발굴해 낸 점이 매우 통찰력 있다. 또한, 단순한 토큰 제거가 아니라 'Keep-Target-Source'라는 비대칭 구조를 도입하여 핵심 정보는 보호하고 덜 중요한 정보는 효율적으로 통합한 설계가 돋보인다. 특히 Mamba의 고유 특성인 '순서 민감성'을 Reordering 단계로 해결함으로써 기존 ViT 기법들의 한계를 명확히 극복하였다.

### 한계 및 논의사항
-   **$\Delta$의 한계**: 실험을 통해 $\Delta_t$가 가장 좋은 지표임을 보였으나, 저자 스스로 언급했듯이 $B_t$ 역시 유의미한 영향을 미친다. 두 지표를 결합한 하이브리드 스코어링 방식이 더 나은 성능을 낼 가능성이 있으며, 이는 향후 연구 과제로 남겨져 있다.
-   **가정**: 본 연구는 재학습이 없는 환경을 가정한다. 만약 소량의 파라미터 튜닝이 허용된다면, $\Delta$ 기반의 중요도 측정 방식을 학습 가능한 형태로 최적화하여 더 높은 압축률에서도 성능을 유지할 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 Vision Mamba의 효율적인 추론을 위해 추가 학습이 필요 없는 토큰 감소 프레임워크 **MTR**을 제안한다. Mamba의 timescale parameter $\Delta$를 토큰 중요도 지표로 활용하고, 중요도에 따른 그룹화 및 유사도 기반 병합과 재정렬 과정을 통해 성능 저하를 최소화하면서 계산량을 줄인다. 실험 결과, ViM-B 모델에서 FLOPs를 40% 절감하면서도 정확도 하락을 1.6%로 억제하는 등 기존 ViT 기반 및 Mamba 기반 기법들보다 뛰어난 효율성을 입증하였다. 이는 향후 실시간 Vision Mamba 응용 서비스 구현에 핵심적인 역할을 할 것으로 기대된다.