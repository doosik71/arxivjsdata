# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment

Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, Jiashi Feng

## 🧩 Problem to Solve

최근 딥 CNN 기반의 시맨틱 분할(semantic segmentation)은 큰 발전을 이루었지만, 다음과 같은 두 가지 주요 문제에 직면해 있습니다:

1. **대규모 주석 데이터의 필요성:** 모델 학습을 위해 픽셀 단위로 주석이 달린 대량의 이미지가 필요하며, 이는 수집하는 데 비용이 많이 듭니다.
2. **미등록 클래스에 대한 일반화 능력 부족:** 학습 과정에서 보지 못한 객체 카테고리에 대해서는 분할 성능이 현저히 저하됩니다.

기존 소수샷(Few-shot) 분할 방법들은 소수의 지원 이미지(support images)에서 지식을 추출한 후, 이를 파라메트릭 모듈(parametric module)에 주입하여 쿼리 이미지(query images)를 분할합니다. 하지만 이러한 방식은 지식 추출과 분할 프로세스를 명확히 분리하지 않아 다음과 같은 단점을 가집니다:

- 분할 모델의 표현이 지원 이미지의 특징과 혼합될 수 있습니다.
- 지원 세트의 주석 정보를 충분히 활용하지 못하여 일반화 성능이 만족스럽지 못합니다.

## ✨ Key Contributions

- **새롭고 효과적인 PANet 제안:** 프로토타입 기반의 비파라메트릭(non-parametric) 거리 학습을 활용하는 소수샷 분할 모델인 PANet(Prototype Alignment Network)을 제안합니다. 이는 파라메트릭 분류 아키텍처를 채택한 기존 방법들과 차별화됩니다.
- **새로운 프로토타입 정렬 정규화(PAR) 도입:** 지원 세트의 지식을 완전히 활용하여 소수샷 학습의 성능을 향상시키기 위한 프로토타입 정렬 정규화(Prototype Alignment Regularization, PAR) 기법을 제안합니다.
- **약한 주석(Weak Annotations)에 대한 적용 가능성:** 스크라이블(scribble)이나 바운딩 박스(bounding box)와 같은 약한 주석을 가진 소수의 예시로부터 직접 학습할 수 있음을 보여줍니다.
- **최고 성능 달성:** PASCAL-5i 데이터셋에서 1-샷 설정(1-shot setting)에서 48.1%, 5-샷 설정(5-shot setting)에서 55.7%의 mIoU(mean Intersection-over-Union) 점수를 달성하여 기존 최신 기술(state-of-the-art)을 1.8%에서 최대 8.6%까지 능가합니다.

## 📎 Related Works

- **시맨틱 분할 (Semantic Segmentation):** FCN [13], SegNet [1], DeepLab [2], PSPNet [29], RefineNet [10] 등 딥 CNN 기반의 방법론들이 발전해왔습니다. 본 연구는 FCN 구조를 따르며, Dilated convolution [27, 2]을 사용하여 수용장(receptive field)을 넓혔습니다.
- **소수샷 학습 (Few-shot Learning):** 메트릭 학습(metric learning) 기반 방법 [25, 23, 24], 최적화 프로세스 학습(learning the optimization process) [18, 6], 그래프 기반 방법 [20, 12] 등이 있습니다. 특히, Prototypical Network [23]는 각 클래스를 하나의 특징 벡터(프로토타입)로 표현하는 방법을 제안했으며, PANet은 이를 조밀한 예측(dense prediction) 작업으로 확장한 형태입니다.
- **소수샷 분할 (Few-shot Segmentation):** Shaban et al. [21]의 OSLSM, Rakelly et al. [16]의 co-FCN, Zhang et al. [28]의 SG-One, Hu et al. [8]의 A-MCG 등이 있습니다. 이들 방법은 주로 파라메트릭 모듈을 사용하여 지원 세트의 정보를 쿼리 분할에 융합합니다. Dong et al. [4]의 PL 또한 프로토타입 네트워크 아이디어를 사용했지만, 3단계 학습과 복잡한 설정으로 인해 본 모델보다 복잡합니다.

## 🛠️ Methodology

PANet은 지원(support) 이미지와 쿼리(query) 이미지를 임베딩 공간(embedding space)에 매핑하고, 각 클래스에 대한 프로토타입을 학습한 후, 각 픽셀을 가장 가까운 프로토타입에 매칭하여 분할을 수행합니다.

1. **문제 설정:**

   - 학습 및 테스트 데이터셋은 서로 겹치지 않는 클래스 집합($C_{seen}$과 $C_{unseen}$)으로 구성됩니다.
   - 각 에피소드는 주석이 있는 지원 이미지 $S$와 쿼리 이미지 $Q$로 이루어집니다.
   - $C$-way $K$-shot 학습을 수행하며, $K$는 클래스당 지원 이미지 수, $C$는 에피소드당 클래스 수입니다.

2. **모델 개요:**

   - **공유 피처 추출기(Shared Feature Extractor):** VGG-16 네트워크를 백본(backbone)으로 사용합니다. `maxpool4` 레이어의 스트라이드를 1로 설정하여 공간 해상도를 유지하고, `conv5` 블록의 컨볼루션을 확장 컨볼루션(dilated convolutions)으로 대체하여 수용장을 늘립니다.
   - **프로토타입 학습 (Prototype Learning):** 지원 이미지 $I_{c,k}$에서 마스크된 평균 풀링(Masked Average Pooling)을 통해 각 클래스의 프로토타입 $p_c$와 배경 프로토타입 $p_{bg}$를 추출합니다. "Late fusion" 전략을 채택하여 특징 맵에 직접 마스크를 적용합니다.
     $$p_c = \frac{1}{K} \sum_{k} \frac{\sum_{x,y} F^{(x,y)}_{c,k} \mathbf{1}[M^{(x,y)}_{c,k}=c]}{\sum_{x,y} \mathbf{1}[M^{(x,y)}_{c,k}=c]}$$
     $$p_{bg} = \frac{1}{CK} \sum_{c,k} \frac{\sum_{x,y} F^{(x,y)}_{c,k} \mathbf{1}[M^{(x,y)}_{c,k} \notin C_i]}{\sum_{x,y} \mathbf{1}[M^{(x,y)}_{c,k} \notin C_i]}$$
     여기서 $F^{(x,y)}_{c,k}$는 특징 맵, $M^{(x,y)}_{c,k}$는 마스크, $\mathbf{1}(\cdot)$은 지시 함수입니다.

3. **비파라메트릭 메트릭 학습 (Non-parametric Metric Learning):**

   - 쿼리 특징 맵 $F_q$의 각 픽셀과 학습된 프로토타입 $P = \{p_c | c \in C_i\} \cup \{p_{bg}\}$ 사이의 거리를 계산합니다.
   - 코사인 거리를 사용하며, 여기에 스케일링 인자 $\alpha=20$을 곱합니다.
   - 거리에 소프트맥스(softmax)를 적용하여 확률 맵 $\tilde{M}_q$를 생성합니다:
     $$\tilde{M}^{(x,y)}_{q;j} = \frac{\exp(-\alpha d(F^{(x,y)}_q,p_j))}{\sum_{p_j \in P} \exp(-\alpha d(F^{(x,y)}_q,p_j))}$$
   - 예측된 분할 마스크 $\hat{M}^{(x,y)}_q = \arg \max_j \tilde{M}^{(x,y)}_{q;j}$를 얻습니다.
   - 분할 손실 $L_{seg}$는 예측된 확률 맵과 실제 쿼리 마스크 $M_q$ 사이의 교차 엔트로피(cross-entropy)로 계산됩니다:
     $$L_{seg} = -\frac{1}{N} \sum_{x,y} \sum_{p_j \in P} \mathbf{1}[M^{(x,y)}_q=j] \log \tilde{M}^{(x,y)}_{q;j}$$

4. **프로토타입 정렬 정규화 (Prototype Alignment Regularization, PAR):**

   - 쿼리 이미지와 예측된 마스크를 새로운 지원 세트로 간주하여 역방향 분할을 수행합니다.
   - 쿼리 특징 맵과 예측 마스크를 사용하여 새로운 프로토타입 $\bar{P}$를 추출합니다.
   - 이 $\bar{P}$를 사용하여 원래의 지원 이미지를 분할하고, 실제 지원 마스크 $M_{c,k}$와 비교하여 손실 $L_{PAR}$을 계산합니다:
     $$\tilde{M}^{(x,y)}_{c,k;j} = \frac{\exp(-\alpha d(F^{(x,y)}_{c,k}, \bar{p}_j))}{\sum_{\bar{p}_j \in \bar{P}} \exp(-\alpha d(F^{(x,y)}_{c,k}, \bar{p}_j))}$$
     $$L_{PAR} = -\frac{1}{CKN} \sum_{c,k,x,y} \sum_{\bar{p}_j \in \bar{P}} \mathbf{1}[M^{(x,y)}_{c,k}=j] \log \tilde{M}^{(x,y)}_{c,k;j}$$
   - 전체 학습 손실은 $L = L_{seg} + \lambda L_{PAR}$이며, 여기서 $\lambda$는 정규화 강도($\lambda=1$ 고정)입니다. PAR은 학습 시에만 적용되며, 추론 시에는 추가적인 계산 비용이 발생하지 않습니다.

5. **약한 주석으로의 일반화:**
   - 본 모델은 스크라이블이나 바운딩 박스와 같은 약한 주석으로도 지원 세트에서 견고한 프로토타입을 추출할 수 있습니다.

## 📊 Results

- **PASCAL-5i 데이터셋 (Mean-IoU):**
  - 1-샷: 48.1% (기존 SOTA 대비 1.8%p 향상)
  - 5-샷: 55.7% (기존 SOTA 대비 8.6%p 향상)
  - 다른 모델들과 달리 1-샷과 5-샷 설정 간의 성능 향상 폭(7.6%p)이 커서, 더 많은 지원 정보를 효과적으로 학습함을 입증했습니다.
- **PASCAL-5i 데이터셋 (Binary-IoU):**
  - 1-샷: 66.5%
  - 5-샷: 70.7%
  - 이전 방법론들을 상회하는 최고 성능을 달성했습니다.
- **PASCAL-5i 데이터셋 (2-way 설정):**
  - 2-way 1-샷 및 2-way 5-샷 분할에서도 이전 연구 대비 20%p 이상 큰 폭으로 뛰어넘는 성능을 보였습니다.
- **MS COCO 데이터셋:**
  - 1-샷: 20.9% (A-MCG 대비 7.2%p 향상)
  - 5-샷: 29.7% (A-MCG 대비 8.2%p 향상)
- **약한 주석 활용:** 스크라이블(44.8% (1-샷), 54.6% (5-샷)) 및 바운딩 박스(45.1% (1-샷), 52.8% (5-샷)) 주석을 사용했을 때도 밀집 주석(dense annotation)과 비교하여 경쟁력 있는 분할 결과를 보여주어 모델의 강건성(robustness)을 입증했습니다.
- **정성적 결과:** 디코더 모듈이나 후처리 기술 없이도 미등록 클래스에 대해 만족스러운 분할 결과를 제공합니다. 외관 변화나 부분만 보이는 객체에 대해서도 견고한 프로토타입을 추출하여 성공적으로 분할합니다.

## 🧠 Insights & Discussion

- **PAR의 효과:** 제안된 프로토타입 정렬 정규화(PAR)는 지원 및 쿼리 프로토타입이 임베딩 공간에서 더 잘 정렬되도록 유도합니다 (PAR 적용 시 평균 Euclidean 거리 32.2 vs. 미적용 시 42.6). 또한, PAR이 모델의 학습 수렴 속도를 가속화하고 더 낮은 손실에 도달하게 돕는다는 것을 학습 손실 곡선을 통해 확인했습니다. 이는 지원 세트의 정보가 더 잘 활용되기 때문입니다.
- **모델의 강점:**
  - 추가적인 학습 가능한 파라미터가 없어서 오버피팅(overfitting)에 덜 취약합니다.
  - 계산된 특징 맵에서 프로토타입 임베딩 및 예측이 수행되므로, 분할에 추가적인 네트워크 통과가 필요하지 않습니다.
  - 정규화가 학습 시에만 적용되므로 추론 비용이 증가하지 않습니다.
  - 약한 주석(스크라이블, 바운딩 박스)에 대해서도 강력한 성능을 보여, 데이터 주석 비용을 절감할 수 있는 가능성을 제시합니다.
- **한계점:**
  - 각 위치에서 독립적으로 예측을 수행하므로 비자연적인 패치(unnatural patches)가 발생할 수 있으나, 이는 후처리를 통해 완화 가능합니다.
  - 임베딩 공간에서 프로토타입이 유사한 의자/테이블과 같은 객체는 구분하기 어려울 수 있습니다.

## 📌 TL;DR

PANet은 심층 CNN의 방대한 주석 데이터 요구 사항과 미등록 클래스에 대한 일반화 능력 부족 문제를 해결하기 위한 소수샷 시맨틱 분할 모델입니다. 이 모델은 메트릭 학습을 기반으로 지원 세트에서 클래스별 프로토타입을 학습하고, 비파라메트릭 거리 계산을 통해 쿼리 이미지를 분할합니다. 특히, 지원 및 쿼리 프로토타입 간의 일관성을 강화하는 **프로토타입 정렬 정규화(PAR)**를 도입하여 지원 정보를 더 효과적으로 활용합니다. 디코더나 후처리 없이도 PASCAL-5i 및 MS COCO 데이터셋에서 1-샷 및 5-샷 설정 모두에서 기존 최고 성능을 크게 뛰어넘었으며, 스크라이블이나 바운딩 박스와 같은 약한 주석에도 강건함을 보여주었습니다.
