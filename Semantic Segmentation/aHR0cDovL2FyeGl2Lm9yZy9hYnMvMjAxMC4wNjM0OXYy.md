# Collaborative Video Object Segmentation by Multi-Scale Foreground-Background Integration

Zongxin Yang, Yunchao Wei, and Yi Yang

## 🧩 Problem to Solve

반지도 비디오 객체 분할(Semi-supervised Video Object Segmentation, VOS)은 첫 프레임의 객체 마스크가 주어졌을 때 비디오 시퀀스 전체에 걸쳐 특정 객체를 분할하는 중요한 컴퓨터 비전 과제입니다. 기존의 최신(SOTA) VOS 방법론들은 주로 전경(foreground) 객체의 임베딩 학습 및 매칭에 중점을 두어, 배경(background) 영역의 특징 임베딩을 간과하는 경향이 있었습니다. 이러한 배경 정보의 무시는 그림 1과 같이 배경에 있는 유사 객체들이 전경 객체와 혼동되는 "배경 혼란(background confusion)" 문제를 야기합니다. 또한, 픽셀 단위 매칭은 배경 픽셀이 전경 픽셀과 유사하게 보일 때 예상치 못한 노이즈를 발생시킬 수 있습니다. 현재 방법들은 종종 방대한 시뮬레이션 데이터에 의존하여 전학습(pre-training)을 수행하며, 이는 학습 과정을 복잡하게 만들고 일반화 능력을 저해할 수 있습니다. 따라서 시뮬레이션 데이터에 의존하지 않으면서도 배경 혼란을 해소하고, 다양한 객체 스케일에 강인하며, 효율적인 VOS 프레임워크가 필요합니다.

## ✨ Key Contributions

* **협업 전경-배경 통합 (Collaborative Foreground-Background Integration, CFBI) 프레임워크 제안:** 전경뿐만 아니라 배경의 특징 임베딩 학습을 동시에 고려하여, 전경과 배경 간의 대조적인 특징을 명시적으로 촉진함으로써 분할 정확도를 향상시킵니다.
* **픽셀 레벨 매칭 및 인스턴스 레벨 어텐션 통합:** CFBI는 픽셀 단위의 세부 정보와 객체 전체의 인스턴스 수준 정보를 모두 활용하여 다양한 객체 스케일에 대한 견고성을 확보합니다. 특히, 다중 지역(multi-local) 매칭을 통해 다양한 객체 이동 속도에 강인하게 대응합니다.
* **효율적인 다중 스케일 매칭 구조 (CFBI+) 도입:** CFBI를 확장하여 Feature Pyramid Network (FPN) 기반의 다중 스케일 특징 맵에서 매칭을 수행함으로써 정확도를 더욱 높입니다.
* **Atrous Matching (AM) 전략 개발:** 매칭 과정의 계산량과 메모리 사용량을 크게 절감하는 플러그 앤 플레이(plug-and-play) 알고리즘으로, $l$ 값을 조절하여 참조 픽셀을 주기적으로 샘플링함으로써 효율성을 증대시킵니다.
* **학습 전략 개선:**
  * **균형 잡힌 무작위 크롭(Balanced Random-Crop):** 전경 픽셀이 충분히 포함된 영역을 무작위로 크롭하여 배경으로의 편향을 방지합니다.
  * **순차 학습(Sequential Training):** 추론 단계와 일관성을 유지하기 위해 학습 중 다음 프레임의 가이던스 마스크를 이전 프레임의 네트워크 예측 결과로 사용합니다.
* **최고 성능 달성:** 시뮬레이션 데이터를 사용하거나 별도의 미세 조정, 후처리 없이 DAVIS와 YouTube-VOS 두 주요 벤치마크에서 기존의 모든 SOTA 방법론들을 능가하는 성능을 달성했습니다.

## 📎 Related Works

* **반지도 비디오 객체 분할 (Semi-supervised Video Object Segmentation):**
  * **테스트 시 미세 조정 기반 (e.g., OSVOS [8], OnAVOS [9], PReMVOS [10]):** 높은 정확도를 달성하지만, 추론 속도가 매우 느리다는 단점이 있습니다.
  * **미세 조정 불필요 기반 (e.g., OSMN [11], PML [27], VideoMatch [28], FEELVOS [12], STMVOS [13]):** 더 빠른 런타임을 목표로 하며, STMVOS [13]는 메모리 네트워크를 사용하여 뛰어난 성능을 보였으나 방대한 시뮬레이션 데이터로 전학습하는 복잡한 과정이 필요합니다. FEELVOS [12]는 픽셀 단위 임베딩 및 전역/지역 매칭 메커니즘을 사용하지만, STMVOS에 비해 성능이 낮습니다.
  * 기존의 픽셀 레벨 매칭 (PML, FEELVOS)은 특징 다양성을 고려하지만, 배경 픽셀이 전경과 유사할 경우 노이즈를 유발할 수 있습니다.
  * 본 논문은 기존 연구들이 전경 특징 임베딩에만 집중한 한계를 극복하기 위해 배경 임베딩 학습을 추가합니다.
* **어텐션 메커니즘 (Attention Mechanisms):**
  * SE-Nets [35]와 같은 채널 어텐션 메커니즘에서 영감을 받아, CFBI는 인스턴스 레벨 평균 풀링을 통해 효율적이고 경량화된 인스턴스 레벨 어텐션 메커니즘을 설계했습니다. 이는 OSMN [11]과 같이 추가적인 컨볼루션 네트워크를 사용하는 방식보다 효율적입니다.

## 🛠️ Methodology

CFBI는 전경과 배경을 동시에 고려하여 객체 분할의 정확도를 높이는 새로운 프레임워크입니다. CFBI+는 여기에 다중 스케일 처리와 Atrous Matching을 추가하여 효율성과 강인함을 더욱 강화합니다.

1. **CFBI 개요 (그림 2 참조):**
    * 백본 네트워크(Dilated ResNet-101 기반 DeepLabv3+)를 사용하여 현재 프레임, 첫 프레임, 이전 프레임의 픽셀 단위 임베딩을 추출합니다.
    * 첫 프레임과 이전 프레임의 임베딩은 마스크를 기반으로 전경과 배경 픽셀로 분리됩니다.
    * **전경-배경 픽셀 레벨 매칭** 및 **협업 인스턴스 레벨 어텐션**을 사용하여 **협업 인셈블러(Collaborative Ensembler, CE)** 네트워크가 예측을 생성하도록 안내합니다.

2. **협업 픽셀 레벨 매칭:**
    * **거리 함수 재설계:** 픽셀 $p$와 $q$ 사이의 거리 $D(p,q)$는 전경 바이어스 $b_{F}$와 배경 바이어스 $b_{B}$를 포함하여 전경/배경 구분을 강화합니다:
        $$ D(p, q) = \begin{cases} 1-\frac{2}{1+\exp(||e_{p}-e_{q}||^{2}+b_{B})} & \text{if } q \in B_{t} \\ 1-\frac{2}{1+\exp(||e_{p}-e_{q}||^{2}+b_{F})} & \text{if } q \in F_{t} \end{cases} $$
    * **전경-배경 전역 매칭 (Global Matching):** 현재 프레임 픽셀 $p$와 첫 참조 프레임 $(t=1)$의 전경 객체 픽셀 집합 $P_{1,o}$ 또는 배경 픽셀 집합 $\bar{P}_{1,o}$ 간의 최소 거리 ($G_{o}(p)$ 및 $\bar{G}_{o}(p)$)를 계산합니다.
    * **전경-배경 다중 지역 매칭 (Multi-Local Matching):** 이전 프레임 $(t=T-1)$과의 지역 매칭 시, 단일 고정 범위가 아닌 여러 이웃 크기 $K=\{k_{1}, \dots, k_{n}\}$ (e.g., $\{4,8,12,16,20,24\}$)를 적용하여 객체의 다양한 이동 속도에 대응합니다. 계산량 증가는 미미합니다.
    * 최종 출력은 현재 프레임의 픽셀 임베딩, 이전 프레임의 임베딩 및 마스크, 다중 지역 매칭 맵, 전역 매칭 맵의 연결(concatenation)로 구성됩니다.

3. **협업 인스턴스 레벨 어텐션 (그림 4 참조):**
    * 첫 프레임과 이전 프레임의 픽셀 임베딩을 전경/배경으로 분리한 후, 각 그룹에 채널별 평균 풀링(channel-wise average pooling)을 적용하여 네 개의 인스턴스 레벨 임베딩 벡터를 생성합니다.
    * 이 벡터들을 연결하여 '협업 인스턴스 레벨 가이던스 벡터'를 생성합니다.
    * 이 가이던스 벡터는 단일 완전 연결(FC) 레이어와 비선형 활성화 함수를 통해 협업 인셈블러 내 Res-Block의 입력 특징 스케일을 채널별로 조절하는 게이트 역할을 합니다. 이는 대규모 객체 분할에 유용합니다.

4. **협업 인셈블러 (CE):**
    * ResNets [39]와 Deeplabs [36]에서 영감을 받아 다운샘플-업샘플 구조를 사용하며, 세 단계의 Res-Block과 Atrous Spatial Pyramid Pooling (ASPP) 모듈을 포함합니다. Dilated Convolution을 사용하여 수용 필드(receptive field)를 효율적으로 확장합니다.

5. **CFBI+: 효율적인 다중 스케일 매칭 (그림 5 참조):**
    * 백본에서 세 가지 스케일(Stride $S=4, 8, 16$)의 특징을 추출하고, FPN [37]을 사용하여 정보를 융합하고 채널 차원을 줄입니다.
    * 각 스케일에서 CFBI의 모든 매칭 프로세스를 수행하며, 각 스케일의 출력은 CE의 해당 단계로 전달됩니다.
    * 큰 스케일에서 작은 스케일로 갈수록 채널 차원을 점진적으로 선형 증가시켜 계산량을 줄이면서 풍부한 의미론적 정보를 활용합니다.
    * **Atrous Matching (AM):**
        * 매칭 프로세스에 Atrous Factor $l$을 도입하여 참조 픽셀을 주기적으로 샘플링합니다 (e.g., $l=2$는 $1/2^2 = 1/4$로 계산 복잡도 감소). 이는 매칭의 효율성을 크게 높입니다.
        * 글로벌 매칭 $G_{o}(p)$는 $G_{l}^{o}(p) = \min_{q \in P_{1,o}^{l}} D(p,q)$로 일반화되며, $P_{1,o}^{l}$은 $l$-atrous 객체 픽셀 집합입니다.
        * CFBI+는 가장 큰 매칭 스케일($S=4$)에 2-atrous 매칭을 적용하여 효율성을 극대화합니다.

6. **구현 상세 및 학습 트릭:**
    * 백본: DeepLabv3+ (Dilated ResNet-101 기반), ImageNet 및 COCO 전학습.
    * **균형 잡힌 무작위 크롭:** 전경 픽셀이 충분히 포함된 시퀀스 프레임을 크롭하여 배경 편향을 완화합니다 (그림 7).
    * **순차 학습:** 학습 중 이전 프레임의 마스크로 네트워크의 이전 예측을 사용하여 추론 단계와 일관성을 유지합니다 (그림 8).
    * 손실 함수: 15%의 가장 어려운 픽셀만을 고려하는 부트스트랩 교차 엔트로피 손실(bootstrapped cross-entropy loss)을 사용합니다.
    * 데이터 증강: 뒤집기(flipping), 스케일링(scaling), 균형 잡힌 무작위 크롭.
    * CE에 Group Normalization (GN) [47] 및 Gated Channel Transformation (GCT) [48]을 적용하여 학습 안정성과 성능을 향상시킵니다.

## 📊 Results

* **YouTube-VOS 벤치마크:**
  * CFBI+는 시뮬레이션 데이터를 사용하거나 미세 조정 없이 82.0% (J&F)의 평균 점수를 달성하며, 모든 다른 SOTA 방법론들을 능가합니다. 2배 강한 학습 스케줄을 적용한 CFBI+ (82.8%)는 STMVOS, KMNVOS 등 시뮬레이션 데이터를 사용한 방법들을 큰 폭(KMNVOS 81.4% 대비 1.4%)으로 뛰어넘습니다.
  * CFBI+는 CFBI보다 더 강인하고 효율적이며(82.0% vs 81.4%, 0.25s vs 0.29s), 특히 보이지 않는(unseen) 카테고리에서 뛰어난 일반화 능력을 보입니다 (테스트 셋에서 78.9% J / 86.8% F).
  * 멀티 객체 추론 속도(4FPS)도 이전 SOTA 방법들보다 훨씬 빠릅니다.

* **DAVIS 2017 벤치마크:**
  * 시뮬레이션 데이터 없이 KMNVOS와 EGMN을 능가하며(CFBI+ 82.9% vs KMNVOS 82.8%), 더 빠른 추론 속도(0.18s vs 0.24s)를 가집니다.
  * 테스트 셋(600p 해상도)에서도 KMNVOS(77.2%)를 0.8% 앞서는 78.0%를 달성하며 강력한 일반화 능력을 입증했습니다.

* **DAVIS 2016 벤치마크:**
  * CFBI+(89.9%)는 시뮬레이션 데이터를 사용하는 KMNVOS(90.5%)와 비슷한 수준의 성능을 달성했으며, FEELVOS(81.7%)에 비해 정확도와 속도 모두에서 월등히 뛰어납니다.

* **어블레이션 연구:**
  * **배경 임베딩:** 배경 임베딩을 제거하면 성능이 74.9%에서 70.9%로 크게 하락하여 전경-배경 협업 학습의 중요성을 입증합니다.
  * **Atrous Matching:** $l=2$ Atrous Matching은 전역 매칭에서 93%의 속도 향상을 가져오면서도 성능 저하는 미미합니다(81.4% vs 81.3%).
  * **다중 스케일 매칭:** 더 큰 스케일에서 매칭을 수행할수록 성능이 향상되지만 계산 비용이 증가합니다. CFBI+는 다중 스케일과 AM을 결합하여 최적의 성능-효율성 균형을 이룹니다.
  * **다른 구성 요소:** 다중 지역 윈도우, 순차 학습, 협업 인셈블러, 균형 잡힌 무작위 크롭, 인스턴스 레벨 어텐션 모두 CFBI 및 CFBI+의 성능 향상에 긍정적으로 기여했습니다.

* **정성적 비교 (그림 9, 10):**
  * CFBI+는 CFBI에 비해 유사한 객체들 사이에서 더 정확한 경계를 생성하며, CFBI가 놓치기 쉬운 작은 객체들도 성공적으로 분할하는 능력을 보여줍니다. 또한, 유사 객체, 작은 객체, 폐색(occlusion) 등 까다로운 VOS 시나리오에서도 강인한 성능을 보입니다.

## 🧠 Insights & Discussion

본 연구는 비디오 객체 분할에서 전경 객체뿐만 아니라 배경 정보의 중요성을 강조하며, 전경-배경 임베딩을 협업적으로 통합하는 새로운 패러다임을 제시합니다. 이러한 접근 방식은 기존 방법론들이 겪었던 배경 혼란 문제를 효과적으로 해소하고, 객체 분할의 정확도를 크게 향상시킵니다. 픽셀 레벨 매칭과 인스턴스 레벨 어텐션을 결합한 하이브리드 접근 방식은 다양한 객체 스케일에 대한 견고성을 제공하며, 동시에 네트워크 구조를 단순하고 빠르게 유지합니다.

특히, CFBI+에서 도입된 다중 스케일 매칭 구조는 VOS 성능을 한 단계 더 끌어올렸으며, Atrous Matching (AM) 전략은 매칭 프로세스의 계산 효율성을 획기적으로 개선하여 고해상도 처리에서도 빠른 속도를 보장합니다. 균형 잡힌 무작위 크롭 및 순차 학습과 같은 간단하면서도 효과적인 학습 기법들은 모델의 학습 안정성과 최종 성능에 크게 기여합니다.

본 연구는 명시적인 한계점을 제시하지는 않지만, 그림 10의 '오토바이' 사례처럼 매우 강한 흐림(blur)이나 극심한 형태 변화가 있는 경우에는 여전히 객체 파트 전체를 정확하게 분할하는 데 어려움이 있을 수 있음을 시사합니다. 그럼에도 불구하고, CFBI와 CFBI+는 시뮬레이션 데이터에 대한 의존도를 없애고 순수 비디오 데이터 학습만으로 SOTA 성능을 달성함으로써, 향후 VOS 연구 및 비디오 객체 추적, 인터랙티브 비디오 편집과 같은 관련 분야의 견고한 기준선 역할을 할 것으로 기대됩니다.

## 📌 TL;DR

반지도 비디오 객체 분할(VOS)은 배경 혼란 문제와 시뮬레이션 데이터에 대한 의존성을 겪습니다. 본 논문은 **전경-배경 통합(CFBI)** 방식을 제안하여 전경 및 배경 임베딩을 대조적으로 학습하고, 픽셀 레벨 매칭과 인스턴스 레벨 어텐션을 결합하여 다양한 객체 스케일에 강인하게 대응합니다. **CFBI+**는 다중 스케일 매칭과 효율적인 **Atrous Matching(AM)** 전략을 추가하여 성능과 속도를 개선합니다. 결과적으로 CFBI+는 시뮬레이션 데이터나 미세 조정 없이 DAVIS 및 YouTube-VOS에서 새로운 최고 성능을 달성하며, 특히 보이지 않는(unseen) 카테고리에서 뛰어난 일반화 능력과 빠른 추론 속도를 보였습니다.
