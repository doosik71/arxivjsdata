# 계층적 전파에서 특징 분리 (Decoupling Features in Hierarchical Propagation)를 통한 비디오 객체 분할

Zongxin Yang, Yi Yang

## 🧩 문제 해결

본 논문은 반지도 학습 비디오 객체 분할(Semi-supervised Video Object Segmentation, VOS)을 위한 계층적 전파 방식의 효율성을 개선하는 데 초점을 맞춥니다. 최근 비전 트랜스포머 기반의 AOT(Associating Objects with Transformers)는 VOS에 계층적 전파를 도입하여 유망한 성능을 보였습니다. AOT의 계층적 전파는 과거 프레임의 정보를 현재 프레임으로 점진적으로 전파하며, 현재 프레임의 특징을 객체 비특이적(object-agnostic)에서 객체 특이적(object-specific)으로 전환합니다.

그러나, 깊은 전파 계층에서 객체 특이적 정보가 증가함에 따라 객체 비특이적 시각 정보의 손실이 불가피하게 발생합니다. 이는 특징의 차원 제약으로 인해 시각 임베딩 학습에 제약을 주며, 더 많은 ID 정보를 포함할수록 AOT의 성능이 저하되는 현상이 관찰됩니다. AOT와 같은 단일 임베딩 공간을 공유하는 방식은 핵심적인 시각 정보를 희석시킬 수 있다는 문제가 있습니다. 본 논문은 이러한 시각 정보 손실 문제를 해결하고 시각 임베딩 학습을 더욱 촉진하는 것을 목표로 합니다.

## ✨ 주요 기여

* **DeAOT 프레임워크 제안:** 계층적 전파에서 객체 비특이적(visual) 특징과 객체 특이적(ID) 특징을 두 개의 독립적인 브랜치로 분리하여 전파하는 Decoupling Features in Hierarchical Propagation (DeAOT) 접근 방식을 제안했습니다. 이를 통해 시각 정보의 손실을 방지하고 VOS 성능을 크게 향상시켰습니다.
* **효율적인 Gated Propagation Module (GPM) 설계:** 듀얼 브랜치 전파로 인한 추가적인 연산 부담을 상쇄하기 위해, 단일 헤드 어텐션(single-head attention)을 기반으로 하는 효율적인 계층적 전파 모듈인 GPM을 설계했습니다. GPM은 기존 AOT의 Multi-head Attention 기반 LSTT 블록보다 더 빠르면서도 효과적입니다.
* **최고 수준의 성능 달성:** YouTube-VOS, DAVIS 2017, DAVIS 2016, VOT 2020의 네 가지 VOS/VOT 벤치마크에서 새로운 SOTA(State-of-the-Art) 성능을 달성했으며, 동시에 뛰어난 런타임 속도를 보여주었습니다. 특히, 테스트 시점 증강(test-time augmentations) 없이도 우수한 결과를 기록했습니다.

## 📎 관련 연구

* **반지도 학습 VOS:** 초기에는 주석된 프레임에 네트워크를 파인튜닝하거나(OSVOS, MoNet, OnAVOS), 픽셀 단위 매칭 맵을 구성하는(PML, FEELVOS, CFBI) 방식이 주를 이루었습니다.
* **어텐션 기반 VOS:** STM, KMN, STCN과 같은 방법들은 메모리 네트워크와 비지역(non-local) 어텐션 메커니즘을 사용하여 과거 프레임의 마스크 정보를 현재 프레임으로 전파했습니다. SST는 트랜스포머 블록의 어텐션 맵을 기반으로 픽셀 단위 매칭을 수행했습니다. AOT는 계층적 전파와 ID(Identification) 메커니즘을 도입하여 여러 객체를 협력적으로 분할하는 방식을 제시했습니다.
* **비전 트랜스포머:** NLP에서 시작된 트랜스포머는 컴퓨터 비전 분야에도 도입되어 이미지 분류, 객체 감지/분할, 비디오 이해 등 다양한 태스크에서 우수한 성능을 보여왔습니다. AOT는 Long Short-Term Transformer (LSTT) 구조를 통해 계층적 전파를 구성했습니다. 본 논문은 AOT와 달리 객체 비특이적 및 객체 특이적 임베딩을 분리합니다.

## 🛠️ 방법론

DeAOT는 비디오 객체 분할을 위해 계층적 전파에서 특징을 분리하는 새로운 프레임워크입니다.
이는 크게 **계층적 듀얼 브랜치 전파**와 **Gated Propagation Module (GPM)**로 구성됩니다.

### 1. 계층적 듀얼 브랜치 전파 (Hierarchical Dual-branch Propagation)

DeAOT는 객체의 시각적 특징과 마스크(ID) 특징을 두 개의 병렬 브랜치에서 전파합니다. 두 브랜치는 동일한 계층 구조를 가지며 $L$개의 전파 계층을 공유합니다.

* **시각 브랜치 (Visual Branch):**
  * 패치 단위 시각 임베딩에 대한 어텐션 맵을 계산하여 객체를 매칭하는 역할을 합니다.
  * 메모리에 저장된 시각 임베딩은 이 어텐션 맵을 사용하여 현재 프레임으로 전파됩니다.
  * 객체 특이적 ID 임베딩 $ \text{ID}(Y_m) $를 직접 활용하지 않으므로, 객체에 편향되지 않고 순수한 시각적 특징을 유지 및 개선하는 데 집중할 수 있습니다.
    $$ \tilde{I}_t^l = \text{Att}(I_t^l W_K^l, I_m^l W_K^l, I_m^l W_V^l) = \text{Corr}(I_t^l W_K^l, I_m^l W_K^l) I_m^l W_V^l $$
    여기서 $I_t^l$은 현재 프레임의 $l$번째 계층 시각 임베딩, $I_m^l$은 메모리 프레임의 시각 임베딩입니다. $W_K^l, W_V^l$은 학습 가능한 가중치 행렬입니다.

* **ID 브랜치 (ID Branch):**
  * 과거 프레임에서 현재 프레임으로 객체 특이적 ID 정보(ID 메커니즘으로 인코딩된 마스크)를 전파하기 위해 설계되었습니다.
  * 객체 식별이 주로 시각적 특징을 기반으로 하기 때문에, 시각 브랜치에서 계산된 동일한 어텐션 맵 $ \text{Corr}(I_t^l W_K^l, I_m^l W_K^l) $를 공유합니다.
    $$ \tilde{M}_t^l = \text{Att}(I_t^l W_K^l, I_m^l W_K^l, M_m^l W_V^l + \text{ID}(Y_m)) = \text{Corr}(I_t^l W_K^l, I_m^l W_K^l) (M_m^l W_V^l + \text{ID}(Y_m)) $$
    여기서 $M_t^l$은 현재 프레임의 $l$번째 계층 객체 특이적 임베딩, $M_m^l$은 메모리 프레임의 객체 특이적 임베딩입니다.

### 2. Gated Propagation Module (GPM)

AOT의 LSTT 블록은 멀티 헤드 어텐션을 사용하여 효율성 병목이 있었습니다. GPM은 이를 개선하여 단일 헤드 어텐션을 기반으로 계층적 전파를 구성합니다.

* **Gated Propagation (GP) 함수:**
  * 표준 어텐션 기반 전파(Eq. 1)를 조건부 게이팅 함수 $ \sigma(U) $와 지역 공간 컨텍스트 모델링을 위한 Depth-wise 2D Convolution $ F_{dw}(\cdot) $로 보강합니다.
    $$ \text{GP}(U, Q, K, V) = F_{dw}(\sigma(U) \odot \text{Corr}(Q, K)V) W_O $$
    여기서 $U$는 게이팅 임베딩, $ \odot $는 요소별 곱셈, $W_O$는 출력 투영 가중치입니다.

* **GPM 구조:**
  * Self-Propagation, Long-term Propagation, Short-term Propagation의 세 가지 종류의 게이티드 전파로 구성됩니다. LSTT와 달리 Feed-forward 모듈을 제거하여 연산량과 파라미터를 절약합니다.
  * **Long-term Propagation:** 메모리 프레임으로부터 정보를 전파합니다 (Eq. 6, 7). ID 전파는 시각 전파의 어텐션 맵을 재사용합니다.
  * **Short-term Propagation:** 이전 프레임의 공간적 근접 영역($ \lambda \times \lambda $)에서 정보를 전파합니다 (Eq. 8, 9). 객체 움직임이 부드럽기 때문에 단거리 전파에서는 비지역 전파가 비효율적임을 감안합니다.
  * **Self-Propagation:** 현재 프레임 내의 객체들을 연결합니다. 어텐션 맵 계산 시 시각 임베딩 $I_t^l$과 ID 임베딩 $M_t^l$을 채널 차원에서 연결($I_t^l \oplus M_t^l$)하여 사용함으로써, ID 임베딩이 시각 임베딩에 대한 위치 임베딩처럼 작동하여 객체 연관을 더 효과적으로 돕습니다 (Eq. 10, 11).

## 📊 결과

DeAOT는 YouTube-VOS, DAVIS 2017, DAVIS 2016, VOT 2020 등 4가지 주요 VOS/VOT 벤치마크에서 기존 AOT 및 다른 SOTA 방법론들을 성능과 효율성 면에서 크게 능가했습니다.

* **YouTube-VOS:** R50-DeAOT-L은 22.4fps에서 86.0% (J&F)의 성능을 달성하여 AOT (14.9fps에서 84.1%)보다 우수했습니다. SwinB-DeAOT-L은 86.2%로 새로운 SOTA를 기록했습니다. 가장 작은 DeAOT-T도 53.4fps에서 82.0%의 성능을 보여, SST보다 빠르면서도 더 높은 정확도를 달성했습니다.
* **DAVIS 2017:** SwinB-DeAOT-L은 유효성 검증 세트에서 86.2%의 SOTA 성능을 보였고, R50-DeAOT-L은 27fps의 실시간 속도로 85.2%를 기록했습니다.
* **DAVIS 2016:** SwinB-DeAOT-L은 92.9%의 J&F 점수로 모든 VOS 방법론을 능가했습니다.
* **VOT 2020:** SwinB-DeAOT-L은 0.622 EAO(Expected Average Overlap)로 MixFormer-L 및 다른 SOTA 트래커를 크게 앞질렀으며, R50-DeAOT-L은 실시간 요구사항 하에 0.571 EAO를 달성했습니다.
* **정성적 결과:** DeAOT는 작거나 크기 변화가 심한 객체에서 AOT보다 나은 성능을 보였지만, 심한 폐색(occlusion) 상황에서 매우 유사한 객체들을 추적하는 데는 여전히 어려움을 겪는 한계도 보였습니다.

**세부 분석 (Ablation Study):**

* **전파 모듈:** 시각 및 ID 임베딩을 분리하지 않으면 성능이 82.5%에서 81.5%로 감소했으며, 채널 차원을 두 배로 늘려도 부분적인 완화만 가능했습니다. GPM 대신 AOT의 LSTT 모듈을 사용하면 성능이 80.3%로 크게 저하되어, 듀얼 브랜치와 GPM 모두 VOS 성능 향상에 필수적임을 확인했습니다.
* **헤드 수:** AOT는 헤드 수가 줄어들면 속도는 빨라지지만 정확도가 하락하는 반면, DeAOT는 제안된 GPM 모듈 덕분에 헤드 수에 강건했습니다.
* **어텐션 맵:** 장기/단기 전파에서 어텐션 맵 구축에 시각 임베딩이 필수적이며, ID 임베딩을 도입하면 오히려 성능이 저하되었습니다 (82.5% vs 82.1%). 반면 자기 전파(self-propagation)에서는 ID 임베딩을 위치 임베딩처럼 활용하는 것이 현재 프레임 내 객체 연관에 도움이 되었습니다 (82.2% vs 82.5%).
* **Depth-wise Convolution ($F_{dw}$)의 커널 크기:** $F_{dw}$가 없으면 DeAOT 성능이 82.5%에서 81.1%로 하락했으며, 최적의 커널 크기는 5로 나타났습니다.

## 🧠 통찰 및 토론

본 연구의 핵심 통찰은 계층적 전파에서 객체 비특이적 시각 정보와 객체 특이적 ID 정보를 혼합하면 중요한 시각적 단서가 희석되어 VOS 성능에 부정적인 영향을 미친다는 것입니다. DeAOT는 이러한 두 가지 정보를 독립적인 브랜치에서 처리함으로써, 시각 브랜치가 순수하게 객체 비특이적 특징을 정교화하여 강력한 매칭 능력을 확보하고, ID 브랜치가 객체 재식별에 필요한 정보를 효율적으로 처리하도록 합니다.

두 브랜치가 어텐션 맵을 공유하는 것은 객체 식별이 주로 시각적 특징에 기반한다는 점을 고려할 때 효율적이고 합리적인 설계입니다. 또한, GPM은 조건부 게이팅과 Depth-wise Convolution을 통해 단일 헤드 어텐션만으로도 강력한 전파 능력을 발휘하며, 이는 듀얼 브랜치 구조의 추가 연산 부담에도 불구하고 전반적인 효율성 향상으로 이어집니다.

한계점으로는 여전히 심한 폐색 상황에서 서로 매우 유사한 객체들을 성공적으로 추적하는 데 어려움이 있다는 점을 들 수 있습니다. 향후 연구에서는 이러한 상황에 대한 강건성을 높이는 방향이 필요할 것입니다.

## 📌 TL;DR

**문제:** 기존 계층적 비디오 객체 분할(VOS) 모델(예: AOT)은 깊은 전파 계층에서 객체 특이적 정보가 증가함에 따라 중요한 객체 비특이적 시각 정보를 손실하여 성능 저하가 발생합니다.

**방법:** DeAOT는 시각 특징과 ID(식별) 특징의 전파를 두 개의 독립적인 브랜치로 분리하고, 두 브랜치 간에 어텐션 맵을 공유합니다. 또한, 효율적인 단일 헤드 어텐션 기반의 Gated Propagation Module (GPM)을 제안하여 듀얼 브랜치로 인한 추가 연산량을 상쇄하고 성능을 높입니다.

**주요 결과:** DeAOT는 여러 VOS/VOT 벤치마크에서 기존 SOTA 모델들을 능가하는 새로운 최고 성능과 뛰어난 런타임 효율성을 달성하여, 특징 분리와 GPM의 효과를 입증했습니다.
