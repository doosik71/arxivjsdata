# Kernelized Memory Network for Video Object Segmentation

Hongje Seong, Junhyuk Hyun, and Euntai Kim

## 🧩 Problem to Solve

반지도 비디오 객체 분할(Semi-supervised Video Object Segmentation, VOS)은 첫 프레임에 주어진 대상 객체의 분할 마스크를 기반으로 이후 프레임에서 해당 객체를 예측하는 작업입니다. 최근 주목받는 Space-Time Memory Networks (STM)는 메모리 네트워크의 비지역적(non-local) 매칭을 활용하지만, VOS 문제는 객체의 움직임이 주로 지역적이라는 점을 간과하여 잘못된 매칭을 유발할 수 있습니다. 또한, 기존 사전 학습 방법은 폐색(occlusion) 상황을 제대로 다루지 못하며, 학습 데이터셋의 객체 경계(segment boundary)가 불분명하거나 부정확하여 모델이 정확한 경계를 학습하기 어렵다는 문제가 있습니다.

## ✨ Key Contributions

* **Kernelized Memory Network (KMN) 제안**: STM의 비지역성(non-locality)을 줄이고 VOS에 메모리 네트워크를 더 효과적으로 적용하기 위해 Gaussian 커널을 도입한 새로운 메모리 읽기(memory read) 작업을 제안했습니다.
* **Hide-and-Seek 전략을 이용한 사전 학습**: 정적 이미지로 KMN을 사전 학습할 때 Hide-and-Seek 전략을 사용하여 폐색에 강건한 예측을 가능하게 하고, 객체 경계를 더 정확하게 추출하도록 돕습니다. 이는 VOS에 Hide-and-Seek를 적용한 첫 번째 시도입니다.
* **최첨단 성능 달성**: DAVIS 2017 test-dev 세트에서 기존 최첨단 방법보다 G$_{M}$ 점수에서 5%p 향상하는 등 주요 벤치마크에서 우수한 성능을 달성했습니다.
* **효율적인 런타임**: STM과 유사한 프레임당 0.12초의 빠른 추론 시간을 유지하면서도 성능을 크게 향상시켰습니다.

## 📎 Related Works

* **반지도 비디오 객체 분할 (Semi-supervised VOS)**: 첫 프레임의 마스크로 후속 프레임을 예측하는 컴퓨터 비전 태스크입니다.
  * **온라인 학습 (Online-learning)**: 테스트 시 주어진 마스크로 네트워크를 미세 조정하며 높은 정확도를 보이지만, 추론 시간이 오래 걸립니다. (예: OSVOS [2], PReMVOS [25])
  * **오프라인 학습 (Offline-learning)**: 고정된 파라미터 세트를 사용하여 훈련하며 빠른 런타임을 제공합니다. (예: RGMP [29], STM [30], FEELVOS [43]) 본 논문은 오프라인 학습 접근 방식을 따릅니다.
* **메모리 네트워크 (Memory Networks)**: Query, Key, Value (QKV) 개념을 사용하여 현재 입력(쿼리)과 과거 입력(메모리) 간의 상관도를 계산하여 가중 평균 값을 검색합니다. STM [30]이 VOS에 성공적으로 적용한 바 있습니다.
* **Kernel Soft Argmax [21]**: 의미론적 대응(semantic correspondence)을 위해 상관 맵에 Gaussian 커널을 적용하여 기울기 전파가 가능한 `argmax` 함수를 만듭니다. 본 연구는 이 아이디어에서 영감을 받았지만, Gaussian 커널의 적용 방식과 목적이 다릅니다.
* **Hide-and-Seek [38]**: 약지도 객체 위치화(weakly supervised object localization)를 개선하기 위해 제안되었으며, 학습 중 객체의 일부를 가려 시스템이 덜 두드러진 부분을 학습하도록 합니다.
* **객체 경계 분할의 어려움**: 이미지 분할에서 객체 경계의 정확한 분할은 여전히 도전적인 과제입니다. EGNet [53]과 LDF [48]와 같은 이전 연구들이 이 문제를 다루었으며, 본 연구는 Hide-and-Seek을 통해 깨끗한 경계를 생성하여 문제를 해결합니다.

## 🛠️ Methodology

1. **아키텍처**: KMN의 전체 아키텍처는 STM [30]과 유사하게 현재 프레임을 쿼리로, 과거 프레임과 해당 마스크를 메모리로 사용합니다. ResNet50 [12]을 사용하여 쿼리와 메모리에서 `key` 및 `value` 피처를 추출하며, 메모리에는 RGB 채널에 마스크를 연결하여 입력합니다.
2. **Kernelized Memory Read**:
    * 기존 STM은 `Query-to-Memory` 매칭만 수행하여 쿼리 프레임의 여러 유사 객체가 메모리 내 동일 대상과 매칭되거나 VOS의 지역성을 간과하는 문제가 있었습니다.
    * KMN은 `Query-to-Memory` 매칭과 `Memory-to-Query` 매칭을 모두 수행합니다.
    * **단계**:
        1. 쿼리 $k_Q(q)$와 메모리 $k_M(p)$ 간의 내적을 통해 상관 맵 $c(p,q) = k_M(p)k_Q(q)^>$를 계산합니다.
        2. `Memory-to-Query` 매칭을 위해 각 메모리 그리드 $p$에 대해 상관 맵에서 가장 잘 매칭되는 쿼리 위치 $\hat{q}(p) = \text{arg max}_{q} c(p,q)$를 찾습니다.
        3. $\hat{q}(p)$를 중심으로 하는 2D Gaussian 커널 $g(p,q) = \text{exp}\left(-\frac{(q_y - \hat{q}_y(p))^2 + (q_x - \hat{q}_x(p))^2}{2\sigma^2}\right)$를 계산합니다 (여기서 $\sigma=7$).
        4. 최종 `Query-to-Memory` 매칭에서 Gaussian 커널을 사용하여 메모리의 `value` $v_M(p)$를 지역적으로 검색합니다. 이 과정은 $r_k(q) = \frac{\sum_p \text{exp}\left(c(p,q)/\sqrt{d}\right) g(p,q) v_M(p)}{\sum_p \text{exp}\left(c(p,q)/\sqrt{d}\right) g(p,q)}$와 같이 표현됩니다.
3. **Hide-and-Seek을 이용한 사전 학습**:
    * 정적 이미지 데이터셋(Pascal VOC, MS COCO 등)으로 KMN을 사전 학습합니다.
    * 단일 이미지를 랜덤 어파인 변환하여 합성 비디오를 생성하고, 추가적으로 이미지의 24x24 그리드 내 특정 셀을 무작위로 숨김으로써 폐색 상황을 인위적으로 만듭니다.
    * 이를 통해 KMN이 **폐색에 강건하게 학습**되며, 동시에 **불분명한 GT 마스크 경계 문제를 해결**하고 더 정확한 객체 경계를 학습하도록 유도합니다.
4. **학습 및 추론 상세**:
    * **학습**: 사전 학습 후 VOS 데이터셋으로 메인 학습을 수행합니다. 동적 메모리 전략과 soft aggregation을 사용하며, `argmax` 함수의 불연속성 때문에 Gaussian 커널은 학습 중에는 적용하지 않습니다.
    * **추론**: 첫 프레임과 이전 프레임은 항상 사용하고, 나머지 중간 프레임은 5프레임 간격으로 선택하는 메모리 관리 전략을 따릅니다. 별도의 테스트 시간 증강(TTA)이나 후처리는 사용하지 않습니다.

## 📊 Results

* **DAVIS 2016 및 DAVIS 2017 유효성 검사 세트**:
  * 정적 이미지로만 사전 학습했을 때, KMN은 기존 STM보다 DAVIS 2016에서 G$_{M}$ 74.8%를 달성하여 7%p 이상 크게 향상되었습니다.
  * DAVIS 데이터셋으로 학습했을 때, DAVIS 2017에서 KMN은 G$_{M}$ 76.0%로 STM의 71.6%를 크게 상회하며 최첨단 성능을 달성했습니다.
  * Youtube-VOS 추가 학습 시, DAVIS 2016 G$_{M}$ 90.5%, DAVIS 2017 G$_{M}$ 82.8%로 모든 기존 VOS 접근 방식 중 가장 우수한 성능을 보였습니다.
* **DAVIS 2017 test-dev 및 Youtube-VOS 2018 유효성 검사 세트**:
  * DAVIS 2017 test-dev에서 KMN은 G$_{M}$ 77.2%를 기록하여 STM의 72.2%보다 5%p 높은 성능으로 최첨단 결과를 달성했습니다.
  * Youtube-VOS 2018 유효성 검사 세트에서도 KMN은 81.4%의 Overall J&F 점수로 최첨단 성능을 유지했습니다.
* **런타임**: DAVIS 2016에서 프레임당 0.12초의 빠른 처리 속도를 보였습니다.
* **정성적 결과**: 빠른 변형, 배경과 유사한 객체, 심한 폐색 등 어려운 상황에서도 일관적으로 정확한 분할 예측을 보여주었습니다. 특히 STM과 비교했을 때, KMN이 폐색과 유사 객체 처리에서 더 강건한 성능을 보였습니다.

## 🧠 Insights & Discussion

* **Kernelized Memory Read의 효과**: KMN의 Kernelized Memory Read는 STM의 비지역적 매칭 문제를 성공적으로 완화하여 VOS에서 메모리 네트워크의 효율성을 높였음을 입증했습니다. 이는 객체의 지역적 연속성을 모델링하는 데 핵심적인 역할을 합니다.
* **Hide-and-Seek의 이점**: Hide-and-Seek 전략을 사전 학습에 적용함으로써 모델이 폐색에 대해 강건해지고, 실제 GT 마스크의 부정확한 경계 문제까지 해결하여 학습된 마스크 경계의 품질을 크게 향상시켰습니다. 픽셀 단위 손실 시각화를 통해 Hide-and-Seek이 생성한 경계에서는 손실이 활성화되지 않아 네트워크가 정확한 경계를 학습했음을 보여줍니다.
* **사전 학습의 중요성**: 정적 이미지와 Hide-and-Seek 전략만을 이용한 사전 학습만으로도 비디오 데이터셋으로 학습된 다른 VOS 방법들과 견줄 만한 성능을 달성하여, 정적 이미지를 VOS 학습에 매우 효율적으로 활용할 수 있음을 입증했습니다.
* **향후 잠재력**: 제안된 Kernelized Memory Network와 Hide-and-Seek 사전 학습 아이디어는 VOS뿐만 아니라 컴퓨터 비전 분야의 다른 분할 관련 작업의 성능 향상에도 기여할 수 있는 큰 잠재력을 가지고 있습니다.

## 📌 TL;DR

KMN (Kernelized Memory Network)은 STM (Space-Time Memory Networks)의 비디오 객체 분할(VOS)에서의 비지역성 문제를 해결하기 위해 Kernelized Memory Read를 제안한다. 이는 2D Gaussian 커널을 도입하여 쿼리-메모리 매칭의 지역성을 강화한다. 또한, 정적 이미지 사전 학습에 Hide-and-Seek 전략을 적용하여 폐색에 강건한 예측과 더 정확한 객체 경계 추출을 가능하게 한다. KMN은 DAVIS 2017 test-dev 및 Youtube-VOS 2018 등 주요 벤치마크에서 기존 최첨단 방법들을 크게 뛰어넘는 성능을 달성하면서도 빠른 추론 시간을 유지한다.
