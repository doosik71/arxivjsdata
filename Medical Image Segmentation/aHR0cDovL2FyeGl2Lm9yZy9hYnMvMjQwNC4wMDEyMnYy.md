# AgileFormer: Spatially Agile Transformer UNet for Medical Image Segmentation

Peijie Qiu, Jin Yang, Sayantan Kumar, Soumyendu Sekhar Ghosh, Aristeidis Sotiras (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 작업에서 대상 객체들이 가지는 **이질적인 외형(Heterogeneous Appearance)**, 즉 다양한 크기와 형태를 효과적으로 처리하지 못하는 기존 Vision Transformer 기반 UNet(ViT-UNet)의 한계이다.

기존의 ViT-UNet 모델들(예: SwinUNet, nnFormer)은 주로 고정된 크기의 윈도우 어텐션(Fixed-sizing window attention)과 고정된 정사각형 패치 임베딩(Rigid square patch embedding)을 사용한다. 이러한 구조는 대상 객체가 고정된 크기의 정사각형 패치 내에 완벽하게 갇혀 있지 않기 때문에, 다양한 크기와 형태를 가진 장기들의 정밀한 특징 표현을 캡처하는 데 어려움이 있다. 결과적으로 이는 특히 크기가 작거나 형태가 불규칙한 장기의 분할 정확도를 떨어뜨리는 원인이 된다.

따라서 본 논문의 목표는 패치 임베딩, 셀프 어텐션, 위치 인코딩(Positional Encoding)이라는 세 가지 핵심 구성 요소에 **공간적 동적 특성(Spatially Dynamic Components)**을 도입하여, 다양한 외형을 가진 객체를 효과적으로 분할할 수 있는 `AgileFormer` 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 설계 아이디어는 ViT-UNet의 전 과정에 걸쳐 **공간적으로 유연한(Spatially Agile)** 메커니즘을 통합하는 것이다. 단순히 어텐션 블록만 개선하는 기존 방식에서 벗어나, 입력 단계부터 특징 추출, 위치 정보 부여 단계까지 모두 동적으로 변화할 수 있도록 설계하였다.

핵심 기여 사항은 다음과 같다:
1. **Deformable Patch Embedding**: 기존의 고정된 정사각형 패치 분할 방식 대신, 학습 가능한 오프셋(Offset)을 통해 유연하게 샘플링하는 디포머블 패치 임베딩을 도입하여 픽셀 수준의 정밀한 로컬라이제이션 능력을 강화하였다.
2. **Spatially Dynamic Self-Attention**: Deformable Multi-head Self-Attention(DMSA)과 Neighborhood Multi-head Self-Attention(NMSA)을 교차적으로 배치하여, 전역적인 문맥 파악과 지역적인 세부 특징 추출을 동시에 달성하였다.
3. **Multi-scale Deformable Positional Encoding (MS-DePE)**: 불규칙하게 샘플링된 그리드에 적합하도록 설계된 새로운 위치 인코딩 방식을 제안하여, 공간적 상관관계를 더욱 정교하게 모델링하였다.

## 📎 Related Works

### CNN 기반 분할 방법
UNet의 등장 이후, 의료 영상 분할에서는 CNN 기반 모델들이 표준으로 자리 잡았다. Attention-UNet, UNet++, nnUNet 등이 대표적이며, 이들은 지역적 특징(Locality)과 이동 불변성(Translation Invariance)을 잘 포착한다. 그러나 CNN은 수용 영역(Receptive Field)이 제한적이기 때문에 전역적인 의미(Global Semantics)를 캡처하는 데 한계가 있다.

### ViT 기반 분할 방법
ViT의 도입으로 전역적 의존성을 캡처할 수 있는 모델들이 등장하였다. TransUNet은 ViT 인코더와 CNN 디코더를 결합하였으나 연산 복잡도가 매우 높았고, SwinUNet은 윈도우 어텐션을 통해 이를 해결하였다. 하지만 본 논문은 이러한 모델들이 여전히 '고정된 크기'의 윈도우와 패치를 사용한다는 점이 한계라고 지적한다.

### 기존 접근 방식과의 차별점
CoTr이나 MERIT 같은 연구들이 공간적으로 가변적인 표현을 시도하였으나, CoTr는 메인 백본이 여전히 CNN이고 디포머블 모듈이 병목(Bottleneck) 지점에만 국한되어 있으며, MERIT는 다해상도 입력을 통해 크기 변화에는 대응하지만 형태의 변화를 캡처하는 데는 한계가 있다. 반면 `AgileFormer`는 순수 ViT-UNet 구조 내에서 패치 임베딩부터 어텐션, 위치 인코딩까지 전 과정을 동적으로 설계하여 차별화를 꾀하였다.

## 🛠️ Methodology

### 전체 시스템 구조
`AgileFormer`는 U-자형 인코더-디코더 구조를 가진 순수 ViT-UNet이다. 인코더와 디코더 모두에서 셀프 어텐션 메커니즘을 통해 특징을 추출하며, 스킵 연결(Skip Connection)을 통해 저수준 특징을 전달한다.

### 주요 구성 요소 및 상세 설명

#### 1. Deformable Patch Embedding
기존 ViT는 이미지를 $n \times n$의 고정된 정사각형 패치로 나누어 1D 벡터로 투영한다. `AgileFormer`는 이를 Deformable Convolution으로 대체한다.
수식적으로 디포머블 컨볼루션은 다음과 같이 정의된다:
$$(f * k)[p] = \sum_{p_k \in \Omega} \phi(f; p + \Delta p + p_k) \cdot k[p_k]$$
여기서 $\Delta p$는 학습 가능한 오프셋이며, $\phi$는 보간 함수(Interpolation function)이다. 본 모델은 두 개의 연속적인 디포머블 컨볼루션 레이어를 사용하여 로컬 표현력을 높였으며, 다운샘플링 단계에서도 겹치는(Overlapping) 커널을 사용하여 지역적 패턴을 보존하였다.

#### 2. Spatially Dynamic Self-Attention
본 모델은 DMSA와 NMSA를 교차적으로 사용하는 블록을 구성한다.

- **Deformable Multi-head Self-Attention (DMSA)**:
  쿼리($Q$)를 통해 오프셋을 생성하고, 이를 통해 불규칙하게 샘플링된 키($\tilde{K}$)와 밸류($\tilde{V}$)를 사용하여 어텐션을 계산한다.
  $$\text{DMSA}_h(f) = \text{softmax}(Q_h \tilde{K}_h^\top / \sqrt{d_k}) \tilde{V}_h$$
  이 방식은 고정된 그리드가 아닌, 객체의 형태에 맞게 동적으로 샘플링 지점을 변경하여 특징을 추출한다.

- **Neighborhood Multi-head Self-Attention (NMSA)**:
  모든 토큰이 아닌, 위치 $p$ 주변의 $k$개 인접 이웃 토큰들과만 어텐션을 계산한다. 이는 연산 복잡도를 선형적으로 줄이면서 CNN과 유사한 지역적 특성을 부여한다.

#### 3. Multi-scale Deformable Positional Encoding (MS-DePE)
DMSA에 의해 샘플링된 그리드가 불규칙하므로, 기존의 절대적/상대적 위치 인코딩은 적용하기 어렵다. 따라서 본 논문은 조건부 위치 인코딩(Conditional PE) 방식을 채택하였다:
$$\text{MS-DePE}(f) = f + P_\theta(f)$$
여기서 $P_\theta$는 서로 다른 커널 크기($3 \times 3$ 및 $5 \times 5$)를 가진 다해상도 디포머블 depth-wise 컨볼루션 레이어로 구현되어, 불규칙한 그리드 상에서도 다각적인 위치 정보를 학습한다.

### 학습 절차 및 손실 함수
모델은 Dice Similarity Coefficient (DSC) 손실과 Cross-Entropy (CE) 손실의 가중 합으로 학습된다:
$$L = \lambda L_{DSC} + (1 - \lambda) L_{cross-entropy}$$
최적화 알고리즘으로는 AdamW를 사용하였으며, 코사인 학습률 감쇠(Cosine learning rate decay)를 적용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: Synapse (다장기 분할), ACDC (심장 MRI), Decathlon (뇌종양 MRI).
- **평가 지표**: Dice Score (DSC $\uparrow$), 95% Hausdorff Distance (HD95 $\downarrow$).
- **비교 대상**: UNet, TransUNet, SwinUNet, nnFormer, MERIT 등 최신 2D 및 3D 모델.

### 주요 결과
- **Synapse (다장기)**:
  - **2D 모델**: AgileFormer-B가 평균 DSC **85.74%**를 달성하여 SOTA를 기록하였다. 특히 대동맥(Aorta)과 췌장(Pancreas) 같이 작고 불규칙한 장기에서 큰 성능 향상을 보였다.
  - **3D 모델**: AgileFormer-T가 평균 DSC **87.43%**, HD95 **7.81mm**를 기록하며 nnFormer를 뛰어넘는 성능을 보였다.
- **ACDC (심장)**: 2D 모델에서 DSC **92.55%**, 3D 모델에서 **92.07%**를 달성하며 기존 모델들보다 우수한 성능을 입증하였다.
- **Decathlon (뇌종양)**: 3D 모델에서 DSC **85.7%**를 기록하여 nnFormer(84.9%) 대비 우위를 점하였다.

### 효율성 및 확장성
AgileFormer는 파라미터 수와 FLOPs 측면에서 SwinUNet보다 약간의 증가(파라미터 $\sim 1.1\%$ 증가, FLOPs $\sim 15\%$ 증가)만 있었음에도 불구하고 훨씬 높은 정확도를 보였다. 특히 모델 크기를 Tiny에서 Base로 확장했을 때 성능 향상 폭이 다른 ViT-UNet보다 훨씬 커서, 뛰어난 확장성(Scalability)을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구의 가장 큰 강점은 **동적 구성 요소의 체계적 도입**이 실제 성능 향상으로 이어진다는 점을 증명한 것이다.
- **디포머블 패치 임베딩의 중요성**: 실험 결과, 고정 패치를 디포머블 패치로 바꾸는 것만으로도 대동맥과 같은 작은 장기의 분할 성능이 비약적으로 향상되었다. 이는 ViT의 고질적 문제인 '지역 정보 손실'을 효과적으로 해결했음을 의미한다.
- **위치 인코딩의 영향**: MS-DePE의 도입은 특히 담낭(Gallbladder) 분할 성능을 4.21% 향상시켰다. 이는 불규칙한 형태의 객체를 인식할 때 적절한 위치 정보 부여가 필수적임을 시사한다.

### 한계 및 비판적 해석
논문에서도 언급되었듯이, 췌장이나 담낭과 같이 **극도로 불규칙하고 구조가 없는(Unstructured)** 객체에 대해서는 여전히 분할 오류가 발생한다. 이는 현재의 디포머블 메커니즘만으로는 해결되지 않는 복잡한 기하학적 변형이 존재함을 의미하며, 향후 더 강력한 공간 가변적 컴포넌트의 연구가 필요함을 보여준다.

또한, 3D 모델의 경우 Tiny 버전만 평가되었는데, Base 이상의 대형 3D 모델에서도 동일한 확장성이 나타날지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

`AgileFormer`는 의료 영상 내 장기들의 다양한 크기와 형태를 처리하기 위해 **Deformable Patch Embedding, Spatially Dynamic Self-Attention, Multi-scale Deformable Positional Encoding**을 통합한 순수 ViT-UNet이다. 이 연구는 고정된 그리드 방식의 한계를 극복하여 2D 및 3D 다장기 분할 작업에서 새로운 SOTA 성능을 달성하였으며, 특히 작은 장기와 불규칙한 형태의 객체 분할 능력을 획기적으로 개선하였다. 향후 다양한 의료 영상 분석 모델 설계 시, 단순한 어텐션 개선을 넘어 입력 단계부터 위치 인코딩까지 전체 파이프라인에 동적 메커니즘을 도입해야 한다는 중요한 설계 방향성을 제시한다.