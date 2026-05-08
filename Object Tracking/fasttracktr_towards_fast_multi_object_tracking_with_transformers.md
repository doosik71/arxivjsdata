# FastTrackTr:Towards Real-time Multi-Object Tracking with Transformers

Liao Pan, Yang Feng\*, Wu Di, Yu Jinwen, Zhao Wenhui, Liu Bo

## 🧩 Problem to Solve

최근 Transformer 기반의 다중 객체 추적(MOT) 모델들이 많은 연구자들의 관심을 받고 있지만, 이 모델들은 주로 구조적 문제나 가변적인 쿼리 개수, 복잡한 네트워크 구조 등으로 인해 추론 속도가 느리다는 고질적인 문제를 안고 있습니다. 이는 실시간 애플리케이션 적용을 어렵게 만듭니다. 본 논문은 과거의 JDT(Joint Detection and Tracking) 패러다임을 재조명하여, 정확하면서도 빠른 실시간 추적을 달성하는 것을 목표로 합니다.

## ✨ Key Contributions

- **FastTrackTr 제안:** 높은 정확도를 유지하면서 실시간 추론 속도를 달성하는 새로운 Transformer 기반 MOT 방법론을 제안합니다.
- **크로스 디코더 메커니즘 도입:** 추가 쿼리나 디코더 없이 과거 궤적 정보를 암묵적으로 통합하는 새로운 크로스 디코더 메커니즘을 소개합니다.
- **광범위한 실험:** DanceTrack, SportsMOT, MOT17 등 여러 벤치마크 데이터셋에서 경쟁력 있는 정확도를 달성하고 추론 속도를 크게 향상시켰음을 입증합니다.

## 📎 Related Works

- **Transformer 기반 MOT:** TransTrack, TrackFormer, MOTR (및 그 개선 버전들: MOTRv2, MeMOTR, MOTRv3, CO-MOT), MOTIP, PuTR 등이 있으며, 주로 추론 속도 문제가 지적되었습니다.
- **Tracking-by-Detection (TBD):** SORT에서 시작하여 DeepSORT, ByteTrack, OC-SORT, HybridSORT 등이 있으며, 주로 탐지 결과와 재식별(Re-ID) 네트워크를 결합하여 추적 정확도를 높였습니다.
- **Joint Detection and Tracking (JDT):** D&T, Integrated-Detection, Tracktor, JDE, FairMOT, CenterTrack, TransTrack 등 과거에는 흔했지만 최근에는 보기 드문 방식입니다. 본 논문은 이 패러다임을 발전시킵니다.

## 🛠️ Methodology

FastTrackTr는 JDT 패러다임을 따르는 Transformer 기반 MOT 프레임워크입니다.

- **FastTrackTr 아키텍처:**
  - **초기 프레임:** 표준 디코더를 사용하여 초기 이력 정보($T_1$)를 초기화합니다.
  - **후속 프레임:** `Historical Encoder`가 이력 정보를 초기화하고, `Historical Decoder` (이력 크로스-어텐션 레이어 포함)가 이 정보를 처리합니다.
  - `ID Embedding head`를 추가하여 객체의 외형 임베딩을 얻고, 이를 탐지 결과와 함께 사용하여 연관 및 매칭을 수행합니다.
  - 백본 네트워크로는 ResNet을 사용하며, 빠른 성능을 위해 RT-DETR을 기본 모델로 채택했습니다.
- **탐지 및 ID 임베딩 학습:**
  - 모델은 단일 순방향 패스에서 타겟의 위치(바운딩 박스)와 외형 임베딩을 동시에 출력합니다.
  - **탐지 손실($L_{det}$):** RT-DETR과 동일하게 L1 손실, 일반화된 IoU 손실, VFL(VariFocalNet) 방식의 IoU 점수를 통합한 분류 손실을 사용합니다.
  - **ID 임베딩 학습($L_{reid}$):**
    - 기존 JDT 방법의 One-Hot 라벨 방식의 한계(차원 폭발, 일반화 제한)를 극복하기 위해 Re-Identification (ReID) 문제로 재정의합니다.
    - `Circle Loss` [35]를 사용하여 동일 객체는 가깝게, 다른 객체는 멀게 임베딩 공간을 최적화합니다. 이는 미세한 식별이 필요한 추적에 적합하며 동적 가중치 부여를 통해 학습을 강화합니다.
  - **총 손실:** $L_{overall} = \lambda_{reid} * L_{reid} + \lambda_{det} * L_{det}$ (여기서 $\lambda_{reid}$와 $\lambda_{det}$는 모두 1로 설정).
- **Historical Decoder 및 Encoder:**
  - **Historical Decoder:** DETR의 셀프-어텐션 메커니즘을 `Historical Cross-Attention`으로 대체합니다.
    - 현재 프레임 쿼리($q_t^d$)와 이전 프레임의 `Historical Encoder` 출력($q_{t-1}'^f$)을 연결($concat(q_t^d, q_{t-1}'^f)$)하여 키와 값으로 사용하고 쿼리는 $q_t^d$를 유지합니다.
    - 이를 통해 이전 프레임의 정보를 현재 프레임의 쿼리에 효율적으로 통합하여 계산 부하를 크게 줄입니다.
  - **Historical Encoder:**
    - 시간적 관계 모델링을 강화하기 위해 도입되었습니다.
    - 마스킹 메커니즘을 사용하여 추적된 객체에 대한 컨텍스트 사전 정보를 제공하고, 관련성이 낮은 이력 요소를 억제합니다.
    - **마스킹 전략:** 훈련 시에는 Ground Truth에 기반하여 마스크를 생성하고 무작위로 일부를 뒤집어 모델 견고성을 높이며, 추론 시에는 객체의 신뢰도 점수를 기반으로 마스크를 결정합니다.
- **매칭 및 연관 (JDE 연관 모듈 개선):**
  - JDE 방식을 따르되, 몇 가지 개선 사항을 적용했습니다.
  - **이중 스테이지 매칭:** ByteTrack의 이중 스테이지 매칭 메커니즘을 도입하여, 먼저 신뢰도 높은 탐지 박스를 연관시킨 후, 낮은 신뢰도 박스를 후보 타겟으로 유지하여 두 번째 라운드에서 매칭합니다.
  - **연관 전략 조정:** 기존 JDE는 ID 특징 기반 연관 후 궤적 기반 연관을 수행했으나, 본 접근 방식은 첫 번째 연관 라운드부터 `외형 비용($A_a$)`과 `움직임 비용($A_m$)`의 가중 합($C=\lambda A_a + (1-\lambda) A_m$)으로 구성된 비용 행렬을 사용합니다.
  - 헝가리안 알고리즘으로 관측치를 추적 체인에 할당하고, Kalman 필터로 궤적을 평활화 및 예측합니다.

## 📊 Results

- **속도 및 정확도 비교:**
  - FastTrackTr는 TensorRT FP16 가속 시 1333x800 해상도에서 86.6 FPS, 640x640 해상도에서 166.4 FPS를 달성하며, 가장 경량화된 `FastTrackTr-R18-Dec3` 버전은 640x640에서 243.4 FPS를 기록했습니다.
  - 이는 MOTIP(RT-DETR) 및 MOTR과 같은 기존 Transformer 기반 모델들보다 월등히 빠른 속도입니다 (최대 약 7배 이상).
  - 동시에 경쟁력 있는 HOTA 및 MOTA 정확도를 유지합니다.
- **SOTA 모델 성능 비교:**
  - **DanceTrack:** FastTrackTr는 62.4 HOTA를 기록하며, 기존 SOTA 모델들 (CMTrack 61.8, C-BIoU 60.6 등)을 능가했습니다.
  - **SportsMOT:** 70.1 HOTA로 MeMOTR (68.8) 및 OC-SORT (68.1)를 넘어섰습니다.
  - **MOT17:** 62.4 HOTA를 기록하며, 이전의 모든 Transformer 기반 모델들을 능가했지만 (PuTR 61.1, MOTIP 59.2), 데이터셋 크기 한계로 인해 일부 CNN 기반 모델보다는 약간 낮았습니다.
- **Ablation 연구:**
  - 제안된 Historical Decoder, Historical Encoder, 마스킹 메커니즘 각각이 모델 성능 향상에 기여함을 확인했습니다.
  - Historical Cross-Attention에 현재 쿼리와 이전 프레임의 쿼리를 `concat`하여 사용하는 방식이 가장 우수했습니다.
  - 훈련 시 8프레임 길이의 비디오 클립을 사용하는 것이 성능과 효율성 사이에서 최적의 균형을 제공했습니다.
  - ID 임베딩 학습에 `Circle Loss`를 사용하는 것이 `Triplet Loss`나 `One-Hot Encoding Loss`보다 월등히 우수한 성능을 보였습니다.
  - JDE 연관 모듈의 개선(바이트트랙 이중 매칭, 비용 행렬 통합)이 추적 성능을 향상시켰습니다.

## 🧠 Insights & Discussion

- **주요 통찰:** FastTrackTr는 Transformer 기반 MOT의 주요 과제였던 속도 문제를 효과적으로 해결하여, 실제 애플리케이션에 적용 가능한 실시간 추적 모델의 가능성을 보여주었습니다. 특히, 확정적(deterministic) 데이터 아키텍처 덕분에 TensorRT 가속화가 원활하여, 동적 연산 그래프를 사용하는 기존 Transformer 모델들의 한계를 극복했습니다.
- **한계점:** MOTIP의 추적 정확도에는 미치지 못한다는 점이 주요 한계로 언급됩니다. 이는 MOTIP가 ID 인코딩을 위한 특수 네트워크 구조와 긴 시퀀스 학습 방법을 사용하여 시간적 정보를 더 잘 학습하기 때문입니다. 하지만 이러한 MOTIP의 장점은 느린 추론 속도와 학습 시간에 대한 트레이드오프가 되므로, FastTrackTr는 실시간 성능과 학습 효율성 측면에서 큰 이점을 가집니다.
- **향후 연구:** 더 복잡한 시나리오에 적응할 수 있도록 모델 구조를 최적화하고, 다양한 하드웨어 플랫폼에서의 배포 가능성을 탐색할 계획입니다.
- **MOTR 추론 속도 문제에 대한 논의 (부록 B):** MOTR 계열 모델의 낮은 GPU 활용률(약 40%)은 동적 쿼리 시스템으로 인한 가변적인 메모리 사용이 원인으로 분석됩니다. PyTorch의 메모리 할당자가 캐시된 블록을 재사용하지 못하고 `cudaMalloc`/`cudaFree` 호출을 빈번하게 발생시켜 상당한 지연을 초래합니다. 쿼리 개수를 고정하고 마스크를 사용하는 방식으로 해결될 수 있음을 시사합니다.

## 📌 TL;DR

Transformer 기반 MOT 모델의 느린 추론 속도 문제를 해결하기 위해, 본 논문은 JDT 패러다임에 기반한 `FastTrackTr`를 제안합니다. 이 모델은 `Historical Encoder`와 `Historical Decoder`의 독창적인 설계를 통해 과거 프레임의 정보를 효율적으로 통합하며, RT-DETR 기반의 빠른 탐지와 `Circle Loss`를 활용한 강건한 ID 임베딩 학습을 특징으로 합니다. 그 결과, FastTrackTr는 SOTA 정확도를 유지하면서도 기존 Transformer 모델보다 훨씬 빠른 실시간 추론 속도를 달성하여 실용적인 다중 객체 추적 솔루션을 제공합니다.
