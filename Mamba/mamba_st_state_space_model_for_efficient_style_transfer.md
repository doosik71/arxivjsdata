# Mamba-ST: State Space Model for Efficient Style Transfer

Filippo Botti et al. (2024)

## 🧩 Problem to Solve

본 논문은 이미지 스타일 전송(Style Transfer) 과정에서 발생하는 막대한 계산 비용 문제를 해결하고자 한다. 스타일 전송은 콘텐츠 이미지의 구조를 유지하면서 스타일 이미지의 예술적 표현(색상, 질감 등)을 입히는 작업이다.

최근의 SOTA(State-of-the-art) 모델들은 주로 Transformer나 Diffusion 기반 모델을 사용한다. 하지만 Transformer는 Self-attention 및 Cross-attention 레이어로 인해 메모리 사용량이 매우 크고 이미지 크기에 따라 계산 복잡도가 이차적으로 증가하는 문제가 있다. 반면 Diffusion 모델은 높은 생성 품질을 보이지만 추론 시간이 매우 길어 실시간 적용에 한계가 있다. 따라서 본 연구의 목표는 이러한 메모리 효율성과 추론 속도 문제를 해결하면서도, 높은 품질의 스타일 전송을 수행할 수 있는 새로운 State Space Model(SSM) 기반의 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 등장한 효율적인 시퀀스 모델인 Mamba의 내부 방정식을 수정하여, Transformer의 Cross-attention과 유사한 기능을 수행하도록 설계한 것이다.

가장 중심적인 기여는 **Mamba-ST Decoder (MSTD)** 블록의 개발이다. 기존 Mamba 모델이 단일 데이터 스트림을 처리하는 것과 달리, MSTD는 스타일 정보와 콘텐츠 정보를 분리하여 입력받고, 이를 SSM의 내부 행렬 조작을 통해 융합한다. 특히 Adaptive Layer Normalization(AdaLN)과 같은 추가적인 정규화 모듈 없이도 SSM의 기본 속성을 유지하며 스타일을 주입할 수 있는 수학적 구조를 설계하였다.

## 📎 Related Works

스타일 전송 분야에서는 AdaIN과 같이 평균과 분산을 조정하는 효율적인 방식부터, Transformer를 이용해 콘텐츠와 스타일 간의 강한 관계를 찾는 고품질 방식, 그리고 Diffusion 모델을 이용한 생성적 방식 등이 연구되었다. Transformer 기반 모델(예: StyTr2)은 품질은 뛰어나나 이미지 크기에 따라 계산량이 급증하는 한계가 있으며, Diffusion 기반 모델(예: StyleID)은 추론 시간이 매우 길다.

SSM 분야에서는 Mamba가 Transformer에 필적하는 성능을 보이면서도 선형 복잡도를 가져 주목받고 있다. 최근 텍스트 기반 스타일 전송에 Mamba를 적용한 시도(StyleMamba)가 있었으나, 이는 Mamba를 단순한 특징 추출기로 사용했을 뿐 스타일 융합을 위해 여전히 AdaLN과 같은 외부 모듈에 의존했다는 한계가 있다. 본 논문은 이러한 외부 모듈 없이 Mamba의 내부 방정식 자체를 수정하여 직접적으로 스타일을 융합한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

Mamba-ST는 크게 세 가지 구성 요소로 이루어진 파이프라인을 가진다.

1. **Mamba Encoders**: 콘텐츠 이미지와 스타일 이미지를 각각 입력받아 시각적 표현을 학습하는 두 개의 인코더이다.
2. **Mamba-ST Decoder (MSTD)**: 두 인코더에서 추출된 특징을 융합하여 스타일이 전송된 표현을 생성하는 핵심 모듈이다.
3. **CNN Decoder**: 융합된 특징 맵을 다시 이미지 형태로 복원하는 역할을 한다.

모든 이미지는 먼저 PatchEmbed 레이어를 통해 1D 임베딩 시퀀스로 변환되어 처리된다.

### Mamba Encoder

인코더는 VMamba 구조를 기반으로 하며, 기울기 소실 문제를 방지하기 위해 레이어 사이에 Skip connection을 추가하였다. 특히 2D 데이터 처리를 위해 네 가지 스캔 방향을 사용하는 **2D-SSM**을 적용하여 공간 정보를 유지한다.

### Mamba-ST Decoder (MSTD)

본 논문의 핵심인 MSTD는 Mamba의 방정식과 Transformer의 Attention 사이의 대칭성($Q \approx C, K \approx B, V \approx X$)에 착안하여 설계되었다.

기존 Mamba의 이산화된 방정식은 다음과 같다.
$$h_k = \bar{A}h_{k-1} + \bar{B}x_k$$
$$y_k = Ch_k + Dx_k$$

Mamba-ST에서는 스타일 이미지($s$)와 콘텐츠 이미지($x$)를 다음과 같이 역할 분담시켜 주입한다.

1. **스타일 의존성**: $\bar{A}, \bar{B}, \Delta$ 행렬을 스타일 소스($s$)로부터 계산한다.
2. **콘텐츠 의존성**: 출력 행렬 $C$를 콘텐츠 소스($x$)로부터 계산한다.
3. **입력 시퀀스**: Selective scan의 입력값으로 콘텐츠가 아닌 스타일 임베딩($s$)을 전달한다.

이를 통해 다음과 같은 융합 과정이 이루어진다.
$$h_k = \bar{A}h_{k-1} + \bar{B}s_k$$
$$y_k = Ch_k$$

이 구조는 내부 상태($h_k$)에 스타일 정보를 저장하고, 이를 콘텐츠 기반의 $C$ 행렬로 변조(Modulation)함으로써 스타일 전송을 수행한다. 또한, 스타일 이미지의 공간적 구조가 그대로 전송되어 콘텐츠와 섞이는 것을 방지하기 위해, 스타일 임베딩에 **Random Shuffle**을 적용하여 고수준의 스타일 정보만 남기고 공간 정보는 제거한다.

### 학습 목표 및 손실 함수

모델은 VGG19 사전 학습 모델을 이용한 Perceptual Loss와 Identity Loss를 사용하여 학습한다.

- **Content Loss ($L^C$)**: 생성 이미지와 원본 콘텐츠 이미지의 특징 간 유클리드 거리를 최소화한다.
- **Style Loss ($L^S$)**: 생성 이미지와 스타일 이미지의 특징 맵의 평균($\mu$)과 표준편차($\sigma$) 차이를 최소화한다.
- **Identity Loss ($L^{id1}, L^{id2}$)**: 콘텐츠나 스타일 이미지 하나만을 입력으로 넣었을 때 자기 자신이 복원되는지를 측정하여 표현 학습을 돕는다.

최종 손실 함수는 다음과 같다.
$$L = \lambda_C L^C + \lambda_S L^S + \lambda_{id1} L^{id1} + \lambda_{id2} L^{id2}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 콘텐츠 이미지는 COCO, 스타일 이미지는 WikiArt 데이터셋을 사용하였다.
- **비교 대상**: StyleID(Diffusion), AesPA-Net, StyTr2(Transformer), AdaAttn, AdaIN.
- **평가 지표**: ArtFID(인간 판단과 높은 상관관계), FID(스타일 유사도), LPIPS 및 CFSD(콘텐츠 보존력).

### 정량적 결과

실험 결과, Mamba-ST는 **ArtFID와 FID 지표에서 SOTA 모델들을 능가**하며 가장 뛰어난 스타일 전송 품질을 보였다. 콘텐츠 보존 지표인 LPIPS와 CFSD에서는 Transformer 기반 모델보다 약간 낮았으나, 전반적인 품질과 보존력의 균형이 가장 우수했다.

효율성 측면에서 Mamba-ST는 Diffusion 기반의 StyleID보다 추론 속도가 훨씬 빠르고 메모리 사용량이 적었으며, Transformer 기반 모델보다 메모리 효율성이 높았다.

### 정성적 결과

시각적 비교 결과, Mamba-ST는 콘텐츠의 색상 일관성을 유지하면서도 스타일을 효과적으로 적용하였다. 특히 Diffusion 모델에서 나타나는 과도한 채도나 대비 현상이 적고, Transformer 모델에서 발생하는 색상 왜곡 문제가 완화된 모습을 보였다.

## 🧠 Insights & Discussion

본 연구는 SSM의 수학적 구조를 변경함으로써 추가적인 모듈 없이도 Cross-attention과 유사한 효과를 낼 수 있음을 증명하였다. 이는 고해상도 이미지 스타일 전송에서 메모리 병목 현상을 해결할 수 있는 매우 효율적인 대안이 된다.

다만, 결과 이미지에서 일부 **패치 간의 불연속성(Gap)**이 발생하는 문제가 관찰되었다. 저자들은 이것이 Mamba가 계승한 RNN의 한계로, 상태 공간 내의 컨텍스트 메모리 저장 방식 때문에 패치 간의 완벽한 연속성을 보장하기 어렵기 때문이라고 분석하였다.

또한, 스타일 임베딩의 Random Shuffle이 필수적임을 확인하였다. 셔플을 제거할 경우, 스타일 이미지의 공간적 구조가 그대로 유지되어 스타일 전송이 아닌 콘텐츠와 스타일 이미지의 단순 합성이 이루어지는 결과가 나타났다.

## 📌 TL;DR

Mamba-ST는 무거운 Transformer의 Attention이나 느린 Diffusion 모델 대신, 효율적인 **State Space Model(SSM)**을 활용한 스타일 전송 프레임워크이다. Mamba의 내부 행렬($A, B, C, \Delta$)에 스타일과 콘텐츠 정보를 분리하여 주입하는 방식을 통해, 적은 메모리와 빠른 속도로 고품질의 스타일 전송을 구현하였다. 이 연구는 향후 실시간 고해상도 스타일 전송 시스템 구축에 중요한 기반이 될 가능성이 크다.
