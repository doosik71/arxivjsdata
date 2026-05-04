# Slot-BERT: Self-supervised Object Discovery in Surgical Video

Guiqiu Liao, Matjaz Jogan, Marcel Hussing, Kenta Nakahashi, Kazuhiro Yasufuku, Amin Madani, Eric Eaton, Daniel A. Hashimoto (2025)

## 🧩 Problem to Solve

본 논문은 수술 비디오에서 레이블이 없는 상태로 객체를 발견(Object Discovery)하고 표현하는 것을 목표로 한다. 수술 비디오는 일반적인 비디오와 달리 기구와 조직의 움직임이 매우 복잡하며, 특정 객체의 가시성이 시간에 따라 크게 변하는 특성을 가진다. 

기존의 객체 중심(Object-centric) 학습 방법들은 크게 두 가지 방향으로 나뉘는데, RNN 기반의 순차적 처리는 효율적이지만 긴 비디오에서 장기적인 시간적 일관성(long-range temporal coherence)을 유지하는 데 어려움이 있다. 반면, 전체 비디오를 병렬로 처리하는 방식은 일관성은 높지만 계산 비용이 매우 커서 의료 현장의 하드웨어에서 구현하기에 비현실적이다. 따라서 본 연구는 **계산 효율성을 유지하면서도 긴 수술 비디오에서 강건한 시간적 일관성을 갖는 자기지도학습(Self-supervised learning) 기반의 객체 표현 학습 모델을 개발**하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오의 각 프레임에서 학습된 **Slot(객체를 대표하는 벡터)들을 마치 자연어 처리의 단어 임베딩처럼 취급하여, 이를 Bidirectional Transformer Encoder로 처리**하는 것이다.

주요 기여 사항은 다음과 같다.
1. **Slot-BERT 아키텍처 제안**: 비디오 프레임 간의 Slot들을 양방향으로 추론하는 Transformer 구조를 도입하여, 긴 비디오 시퀀스에서도 효율적으로 객체 중심 표현을 학습한다.
2. **Slot Contrastive Loss 도입**: Slot 간의 직교성(Orthogonality)을 높여 latent space에서 각 Slot이 서로 중복되지 않고 독립적인 객체 개념을 담도록 강제함으로써 표현의 분리(Disentanglement) 성능을 향상시켰다.
3. **범용적 성능 및 제로샷 적응력 입증**: 복부, 담낭 절제술, 흉부 수술 등 서로 다른 도메인의 4개 수술 데이터셋에서 기존 SOTA 모델보다 뛰어난 성능을 보였으며, 학습하지 않은 도메인에 대해서도 효율적인 제로샷(Zero-shot) 적응 능력을 입증하였다.

## 📎 Related Works

### 1. 자기지도 객체 중심 학습 (Self-supervised Object-centric Learning)
최근 Slot Attention (SA) 메커니즘을 통해 저수준의 픽셀 데이터를 고수준의 객체 단위 Slot으로 그룹화하는 연구가 활발하다. 하지만 비디오 데이터의 경우, 시간축으로 Slot을 연결하기 위해 주로 RNN 기반 구조를 사용하는데, 이는 긴 시퀀스에서 학습 불안정성과 일관성 결여라는 한계가 있다.

### 2. 마스크 정보 인코딩 (Masked Information Encoding)
BERT나 MAE와 같이 입력 데이터의 일부를 마스킹하고 이를 복원하는 방식으로 강건한 표현을 학습하는 방법론이 제안되었다. 본 논문은 이러한 아이디어를 픽셀 수준이 아닌, 이미 추출된 'Slot' 수준에서 적용하여 시간적 추론 능력을 극대화하였다.

### 3. 비지도 객체 검출 및 분할 (Unsupervised Object Detection & Segmentation)
기존의 많은 방법론이 Optical Flow와 같은 움직임 큐(Motion cues)에 의존한다. 그러나 Optical Flow는 정적인 객체나 변형 가능한 조직, 또는 조명 변화가 심한 수술 환경에서 오류가 많다. 본 논문은 이러한 외부 큐에 의존하지 않고 특성 복원(Feature reconstruction) 목표만을 사용하여 보다 단순하고 강건한 프레임워크를 구축하였다.

## 🛠️ Methodology

### 전체 파이프라인
Slot-BERT의 전체 구조는 **ViT Encoder $\rightarrow$ Recurrent Slot Attention $\rightarrow$ Temporal Slot Transformer (TST) $\rightarrow$ Slot Decoder** 순으로 구성된다.

1. **특성 추출**: 입력 비디오 프레임 $I_t$를 Vision Transformer (ViT) Encoder에 통과시켜 패치 기반의 특성 맵 $X \in \mathbb{R}^{N \times D_{feature} \times T}$를 얻는다.
2. **초기 Slot 생성**: Recurrent Slot Attention 모듈을 통해 각 프레임의 특성을 $K$개의 Slot $s_t \in \mathbb{R}^{K \times d_{slot}}$으로 그룹화한다. 이때 $t$번째 프레임의 초기 Slot은 $t-1$번째 프레임의 최종 Slot 값을 사용하여 시간적 연속성을 부여한다.
3. **Temporal Slot Transformer (TST)**: 
    - 학습 가능한 Temporal Positional Embedding을 더해 시간 정보를 부여한다.
    - **Masked Training**: 학습 시 특정 비율 $\gamma$로 Slot들을 무작위로 마스킹($0$으로 설정)한다.
    - 마스킹된 Slot 시퀀스를 Bidirectional Transformer Encoder에 통과시켜, 주변 프레임의 정보를 이용해 마스킹된 부분을 예측하고 시간적으로 융합된 최종 Slot $S_{final}$을 생성한다.
4. **복원 및 디코딩**: 최종 Slot들을 Slot Decoder(MLP Broadcast 또는 SlotMixer)를 통해 다시 원래의 특성 공간 $X_{recon}$으로 복원한다.

### 주요 방정식 및 손실 함수

#### 1. Slot Attention 업데이트
Slot $s$는 다음과 같은 iterative dot-product attention을 통해 업데이트된다.
$$\hat{A}_{ij} = \frac{A_{ij}}{\sum_{l=1}^{N} A_{il}}, \quad A = \text{softmax}\left(\frac{q k^T}{\sqrt{d}}\right)$$
여기서 $q$는 Slot의 쿼리, $k$와 $v$는 이미지 특성의 키와 값이다.

#### 2. Slot Contrastive Loss
Slot 간의 중복을 줄이고 독립성을 높이기 위해 코사인 유사도 기반의 대조 학습 손실을 사용한다.
$$\text{sim}(u_i, u_j) = \frac{u_i \cdot u_j}{\|s_i\|\|s_j\|}$$
자기 유사도를 제외한 유사도 행렬 $C_{ij} = \text{sim}(u_i, u_j) - \delta_{ij}$를 이용하여 다음과 같이 손실 함수를 정의한다.
$$L_{contrast} = \frac{1}{T \cdot K^2} \sum_{t=1}^{T} \sum_{i=1}^{K} \sum_{j=1}^{K} \left[ -\log \frac{\exp(-C_{ij}/\tau)}{\sum_{k=1}^{K} \exp(-C_{ik}/\tau)} \right]$$
이 손실 함수는 각 Slot 벡터들이 latent space에서 서로 직교하도록 유도한다.

#### 3. 최종 학습 목표
모델은 특성 복원 손실($L_{recon}$)과 대조 손실($L_{contrast}$)의 가중 합을 최소화하도록 학습된다.
$$L_{final} = \|X_{recon} - X\|_2^2 + \alpha L_{contrast}$$

## 📊 Results

### 실험 설정
- **데이터셋**: MICCAI (동물/팬텀), Cholec80 (담낭 절제술), EndoVis 2017 (복부 수술), Thoracic (흉부 수술) 등 4개 데이터셋을 사용하였다.
- **지표**: 객체 분할 및 위치 정확도를 측정하기 위해 mBO-V(비디오 수준 overlap), mBO-F(프레임 수준 overlap), FG-ARI(분할 유사도), mBHD(경계 거리), CorLoc(위치 정확도)를 사용하였다.
- **비교 대상**: SAVi, STEVE, DINOSaur, Video-Saur, Slot-Diffusion 등 최신 객체 중심 학습 모델들과 비교하였다.

### 주요 결과
1. **비지도 학습 성능**: MICCAI 데이터셋에서 mBO-V 48.90%, FG-ARI 58.20%를 기록하며 모든 지표에서 SOTA를 달성하였다. 특히 Video-Saur 대비 mBHD와 CorLoc에서 큰 개선을 보여 경계 및 인스턴스 정확도가 높음을 입증하였다.
2. **전이 학습 및 제로샷 성능**: MICCAI에서 학습한 모델을 Cholec 데이터셋에 미세 조정(Fine-tuning)했을 때 성능이 크게 향상되었으며, 전혀 학습하지 않은 EndoVis 및 Thoracic 데이터셋에서도 타 모델 대비 압도적인 제로샷 분할 성능을 보였다.
3. **긴 시퀀스 대응 능력**: 프레임 수를 5에서 11까지 늘렸을 때, 타 모델들은 mBO-V(시간적 일관성)가 급격히 하락했으나, Slot-BERT는 매우 완만하게 하락하며 강건한 시간적 추론 능력을 보여주었다.
4. **데이터 효율성**: 학습 데이터 양을 1%까지 줄였을 때도 Video-Saur(6.0%) 대비 훨씬 높은 mBO-V(19.6%)를 유지하여 데이터 부족 상황에서도 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 픽셀 레벨의 무거운 연산 대신, 압축된 Slot 임베딩 상에서 Transformer의 양방향 추론을 수행함으로써 **계산 효율성과 시간적 일관성이라는 두 마리 토끼를 모두 잡았다**. 특히 외부 큐(Optical Flow 등) 없이 오직 자기지도 학습만으로 수술 도구를 정확히 분리해낸 점이 고무적이다.

### 한계 및 비판적 해석
- **경계 정밀도 부족**: 패치 기반 처리($P \times P$)를 수행하기 때문에, 객체의 대략적인 위치와 형태는 잘 잡지만 픽셀 단위의 정밀한 경계(pixel-level boundary)를 찾아내는 데는 한계가 있다. 이는 수술 로봇의 정밀 제어에 적용하기 위해서는 추가적인 고해상도 정제 과정이 필요함을 시사한다.
- **Slot 개수($K$)의 고정**: 본 모델은 $K$값을 고정하여 사용하는데, 실험 결과 $K=7$에서 최적의 성능을 보였다. 하지만 실제 수술 상황에서는 나타나는 도구의 개수가 가변적이므로, 이를 동적으로 조절하는 Dynamic Slot Allocation 기법이 추가될 필요가 있다.

## 📌 TL;DR

Slot-BERT는 수술 비디오의 객체들을 Slot이라는 벡터로 표현하고, 이를 **양방향 Transformer**로 처리하여 시간적 일관성을 극대화한 비지도 학습 모델이다. **Slot Contrastive Loss**를 통해 객체 간 표현의 독립성을 높였으며, 이를 통해 계산 비용을 낮추면서도 긴 비디오에서 뛰어난 객체 발견 및 분할 성능을 달성하였다. 특히 제로샷 적응 능력이 뛰어나 다양한 수술 도메인에 유연하게 적용 가능하며, 향후 수술 보조 시스템의 자동화 및 주석 작업의 효율화를 위한 기초 기술로 활용될 가능성이 높다.