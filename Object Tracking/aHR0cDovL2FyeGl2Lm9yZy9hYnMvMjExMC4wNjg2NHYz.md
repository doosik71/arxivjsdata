# ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang

## 🧩 Problem to Solve
기존의 다중 객체 추적(MOT) 방법들은 탐지 점수(detection score)가 낮은 객체 탐지 결과(detection box)들을 단순히 버립니다. 이는 종종 가려진(occluded) 객체와 같은 실제 객체의 누락 및 궤적 단편화(fragmented trajectories)를 야기하여, MOT 성능에 돌이킬 수 없는 오류를 발생시킵니다. 본 논문은 이러한 저점수 탐지 박스를 효과적으로 활용하여 참 객체를 복구하고 배경을 필터링하는 문제를 해결하고자 합니다.

## ✨ Key Contributions
*   **BYTE(BYTE)**라는 간단하고 효과적이며 일반적인 데이터 연관(data association) 방법을 제안했습니다. 이 방법은 모든 탐지 박스를 활용하며, 특히 저점수 탐지 박스를 기존 트랙릿(tracklet)과의 유사성을 이용하여 참 객체를 복구하고 배경을 제거합니다.
*   BYTE를 9가지 최신 추적기에 적용했을 때, IDF1 점수에서 1점에서 10점까지 일관된 성능 향상을 달성했습니다.
*   제안된 BYTE 연관 방법을 고성능 YOLOX 탐지기와 결합한 강력한 추적기 **ByteTrack**을 개발했습니다.
*   ByteTrack은 MOT17 테스트 세트에서 80.3 MOTA, 77.3 IDF1, 63.1 HOTA를 30 FPS의 속도로 달성하며 최초로 최고 성능을 기록했습니다.
*   MOT20, HiEve, BDD100K 벤치마크에서도 최신 기술(SOTA) 성능을 달성했습니다.
*   소스 코드, 사전 학습된 모델, 배포 버전 및 다른 추적기에 적용하는 튜토리얼을 공개했습니다.

## 📎 Related Works
*   **MOT에서의 객체 탐지:** DPM, Faster R-CNN, SDP와 같은 탐지기를 기반으로 하거나, RetinaNet, CenterNet, YOLO 시리즈와 같은 강력한 탐지기를 활용합니다. 비디오 객체 탐지(VOD)는 이전 프레임 정보를 활용하여 탐지 성능을 높이며, 일부 방법은 SOT(Single Object Tracking) 또는 칼만 필터(Kalman Filter)를 사용하여 트랙릿 위치를 예측하고 탐지 결과를 강화합니다. 대부분의 MOT 방법은 저점수 탐지 박스를 제거하지만, 본 논문은 이것이 가려진 객체를 제거하는 문제로 이어진다고 지적합니다.
*   **데이터 연관:**
    *   **유사성 측정:** SORT($\text{SORT}$)는 칼만 필터를 이용해 위치/모션 유사성(IoU)을 사용하며, DeepSORT($\text{DeepSORT}$)는 Re-ID 특징 기반 외관 유사성을 추가하여 장거리 연관을 돕습니다.
    *   **매칭 전략:** 헝가리안 알고리즘(Hungarian Algorithm)이나 그리디 할당(greedy assignment)을 사용합니다. DeepSORT($\text{DeepSORT}$)는 계단식 매칭(cascaded matching)을, MOTDT($\text{MOTDT}$)는 외관 유사성을 먼저 사용합니다. 최근에는 트랜스포머(Transformer) 기반 방법이 암묵적으로 연관을 수행하기도 합니다.
    *   기존 방법들이 더 나은 연관 방법 설계에 집중하는 반면, 본 논문은 탐지 박스를 최대한 활용하는 방식이 데이터 연관의 상한선을 결정한다고 주장합니다.

## 🛠️ Methodology
BYTE는 모든 탐지 박스를 고점수와 저점수 박스로 분리하여 두 단계의 연관 과정을 수행합니다.

1.  **탐지 및 분리:**
    *   주어진 객체 탐지기($\text{Det}$)를 사용하여 현재 프레임($f_k$)의 모든 탐지 박스와 점수를 예측합니다.
    *   사전 정의된 임계값($\tau$)을 기준으로 탐지 박스들을 **고점수 탐지 박스**($D_{high}$)와 **저점수 탐지 박스**($D_{low}$)로 분리합니다.

2.  **트랙릿 위치 예측:**
    *   칼만 필터(Kalman Filter)를 사용하여 현재 프레임에서 기존 트랙($T$)의 새로운 위치를 예측합니다.

3.  **첫 번째 연관 (고점수 탐지 박스):**
    *   모든 트랙($T$)과 고점수 탐지 박스($D_{high}$) 사이에 첫 번째 연관을 수행합니다.
    *   유사성($\text{Similarity\#1}$)은 IoU(Intersection over Union) 또는 Re-ID(Re-identification) 특징 거리로 계산할 수 있습니다.
    *   헝가리안 알고리즘을 사용하여 매칭을 완료합니다.
    *   매칭되지 않은 고점수 탐지 박스는 새로운 트랙($D_{remain}$)으로 초기화될 후보가 되고, 매칭되지 않은 트랙($T_{remain}$)은 두 번째 연관의 대상이 됩니다.

4.  **두 번째 연관 (저점수 탐지 박스):**
    *   첫 번째 연관 후 남은 트랙($T_{remain}$)과 저점수 탐지 박스($D_{low}$) 사이에 두 번째 연관을 수행합니다.
    *   유사성($\text{Similarity\#2}$)은 **IoU만을 사용합니다.** 이는 저점수 탐지 박스가 심각한 가림(occlusion)이나 모션 블러(motion blur)를 포함하는 경우가 많아 Re-ID 특징이 신뢰할 수 없기 때문입니다.
    *   이 단계에서 매칭되지 않은 저점수 탐지 박스들은 배경으로 간주되어 삭제됩니다.

5.  **트랙 관리:**
    *   두 번째 연관 후에도 매칭되지 않은 트랙($T_{re-remain}$)은 일정 프레임 수(예: 30프레임) 이상 손실되면 삭제되고, 그렇지 않으면 손실된 트랙($T_{lost}$)으로 유지됩니다 (재탄생 과정).
    *   첫 번째 연관에서 매칭되지 않은 고점수 탐지 박스($D_{remain}$)로부터 새로운 트랙을 초기화합니다.

ByteTrack은 고성능 YOLOX 탐지기와 위에서 설명한 BYTE 연관 방법을 결합하여 구성됩니다.

## 📊 Results
*   **BYTE의 효과성:**
    *   **유사성 분석:** 첫 번째 연관에서는 IoU 또는 Re-ID 모두 적합하지만, 두 번째 연관에서는 저점수 탐지 박스의 신뢰할 수 없는 Re-ID 특징 때문에 IoU가 필수적임을 확인했습니다.
    *   **기존 연관 방법과의 비교:** SORT($\text{SORT}$), DeepSORT($\text{DeepSORT}$), MOTDT($\text{MOTDT}$)와 비교하여 MOTA, IDF1, IDs(identity switches)에서 상당한 개선을 보였습니다. 특히 IDs를 크게 줄였습니다 (SORT($\text{SORT}$) 대비 IDs 291에서 159로 감소). 저점수 탐지 박스에서 더 많은 참 양성(TPs)을 복구하여 MOTA 향상에 기여했습니다.
    *   **임계값($\tau$)에 대한 강건성:** 기존 SORT($\text{SORT}$)보다 탐지 점수 임계값 변화에 더 강건한 성능을 보였습니다.
    *   **다른 추적기에 적용:** JDE($\text{JDE}$), FairMOT($\text{FairMOT}$), CenterTrack($\text{CenterTrack}$), TransTrack($\text{TransTrack}$) 등 9개 최신 추적기에 BYTE를 적용했을 때 MOTA, IDF1, IDs에서 일관된 성능 향상을 보였습니다.

*   **ByteTrack 벤치마크 평가:**
    *   **MOT17:** 80.3 MOTA, 77.3 IDF1, 63.1 HOTA, 30 FPS로 모든 추적기 중 1위를 기록했습니다. 더 적은 훈련 데이터를 사용했음에도, 복잡한 Re-ID나 어텐션(attention) 메커니즘 없이 단순한 칼만 필터 기반 유사성만으로 최고 성능을 달성했습니다.
    *   **MOT20:** 77.8 MOTA, 75.2 IDF1, 61.3 HOTA로 1위를 기록했으며, 특히 ID 스위치를 71% 감소시켜 혼잡한 상황과 가림에 매우 강건함을 입증했습니다.
    *   **HiEve:** 61.7 MOTA, 63.1 IDF1로 1위를 기록하여 복잡한 이벤트와 다양한 카메라 뷰에 대한 강건성을 보였습니다.
    *   **BDD100K:** 45.5 mMOTA, 54.8 mIDF1로 1위를 기록하여 낮은 프레임률과 큰 카메라 움직임이 있는 자율주행 장면에서도 뛰어난 성능을 보였습니다.
*   **추가 분석:**
    *   **경량 모델 적용:** 경량 YOLOX($\text{YOLOX}$) 백본(Nano, Tiny)을 사용하더라도 DeepSORT($\text{DeepSORT}$) 대비 MOTA와 IDF1에서 안정적인 개선을 보여 실제 응용에서의 매력을 높였습니다.
    *   **훈련 데이터:** Mosaic($\text{Mosaic}$) 및 Mixup($\text{Mixup}$)과 같은 강력한 데이터 증강 덕분에 많은 양의 훈련 데이터 없이도 높은 성능을 달성했습니다. CrowdHuman($\text{CrowdHuman}$) 데이터셋 추가는 가려진 사람 탐지 능력을 향상시켜 IDF1을 크게 높였습니다.
    *   **트랙릿 보간(Tracklet Interpolation):** 후처리 단계로 완전히 가려진 객체를 처리하여 MOTA 및 IDF1을 추가로 개선했습니다.

## 🧠 Insights & Discussion
ByteTrack은 저점수 탐지 박스에 대한 기존의 통념을 깨고, 이들이 객체 추적에 있어 중요한 정보를 담고 있음을 성공적으로 입증했습니다. 특히 가림 현상이 빈번한 경우, 저점수 탐지 박스를 적절히 활용함으로써 누락된 객체를 복구하고 궤적의 연속성을 유지하는 데 결정적인 역할을 한다는 점이 핵심 통찰입니다.

이 연구는 탐지 결과의 "공정성(fairness)"이라는 관점에서 MOT를 재해석하며, 탐지기와 연관 방법의 접점에서 발생하는 문제를 효과적으로 해결했습니다. ByteTrack의 단순성, 높은 정확도, 빠른 처리 속도는 복잡한 실제 환경에서의 다중 객체 추적 애플리케이션에 매우 매력적인 솔루션이 될 수 있음을 시사합니다. 한계점으로는 빠른 비선형 움직임이 많은 상황에서 Re-ID 특징 없이 IoU 기반 매칭만으로 충분한지, 그리고 $\tau$ 임계값 선택의 영향은 여전히 존재한다는 점 등이 있을 수 있습니다.

## 📌 TL;DR
기존 MOT($\text{MOT}$)가 저점수 탐지 박스를 버려 객체 누락과 궤적 단편화가 발생한다는 문제에 대응하여, **BYTE**는 고점수와 저점수 탐지 박스를 모두 활용하는 2단계 연관 방법을 제안합니다. 특히, 저점수 박스를 기존 트랙릿과 IoU($\text{IoU}$) 기반으로 재연관하여 가려진 객체를 복구하고 배경을 필터링합니다. 이 방법은 여러 추적기에서 일관된 성능 향상을 보였으며, 이를 YOLOX($\text{YOLOX}$)와 결합한 **ByteTrack**은 MOT17($\text{MOT17}$), MOT20($\text{MOT20}$), HiEve($\text{HiEve}$), BDD100K($\text{BDD100K}$) 등에서 속도와 정확도 모두 SOTA($\text{SOTA}$)를 달성하며, 특히 가림에 강건한 실용적인 MOT 솔루션임을 입증했습니다.