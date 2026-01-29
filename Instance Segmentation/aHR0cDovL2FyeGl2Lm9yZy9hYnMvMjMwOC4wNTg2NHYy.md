# The Multi-modality Cell Segmentation Challenge: Towards Universal Solutions

Jun Ma, Ronald Xie, Shamini Ayyadhury, Cheng Ge, Anubha Gupta, Ritu Gupta, Song Gu, Yao Zhang, Gihun Lee, Joonkee Kim, Wei Lou, Haofeng Li, Eric Upschulte, Timo Dickscheid, José Guilherme de Almeida, Yixin Wang, Lin Han, Xin Yang, Marco Labagnara, Vojislav Gligorovski, Maxime Scheder, Sahand Jamal Rahi, Carly Kempster, Alice Pollitt, Leon Espinosa, Tˆam Mignot, Jan Moritz Middeke, Jan-Niklas Eckardt, Wangkai Li, Zhaoyang Li, Xiaochen Cai, Bizhe Bai, Noah F. Greenwald, David Van Valen, Erin Weisbart, Beth A. Cimini, Trevor Cheung, Oscar Brück, Gary D. Bader, and Bo Wang

## 🧩 Problem to Solve

세포 분할은 현미경 이미지에서 정량적인 단일 세포 분석을 위한 필수 단계입니다. 그러나 기존 세포 분할 방법은 특정 이미지 모달리티에 맞춰져 있거나, 다양한 실험 설정에서 수동으로 하이퍼파라미터를 조정해야 하는 경우가 많습니다. 이는 세포 기원, 현미경 유형, 염색 기술 및 세포 형태의 다양성으로 인해 일반적이고 자동화된 분할 기술 개발을 어렵게 합니다. 또한, 기존 챌린지들은 제한된 현미경 이미지 유형에만 초점을 맞추고 주로 정확도에만 집중하여 효율성을 간과하는 경향이 있어, 개발된 알고리즘의 일반화 가능성과 실용적인 배포에 한계가 있었습니다.

## ✨ Key Contributions

* **대규모 다중 모달리티 벤치마크 데이터셋 구축:** 50개 이상의 다양한 생물학적 실험에서 파생된 1,500개 이상의 라벨링된 이미지로 구성된 포괄적인 다중 모달리티 세포 분할 벤치마크를 제시했습니다. 이는 기존 데이터셋보다 다양성과 규모가 크게 확장되었습니다.
* **범용성 높은 Transformer 기반 알고리즘 개발:** 챌린지 참가자 중 최상위 팀들이 개발한 Transformer 기반 딥러닝 알고리즘은 기존 방법을 능가하며, 수동 파라미터 조정 없이 다양한 이미징 플랫폼 및 조직 유형에 걸쳐 다양한 현미경 이미지에 적용 가능함을 입증했습니다.
* **정확도와 효율성의 균형:** 본 챌린지는 일반화 가능성뿐만 아니라 알고리즘의 효율성(실행 시간, GPU 메모리 사용량)까지 평가하여 실제 생물학 연구 현장에서의 적용 가능성을 높였습니다.
* **오픈소스 및 접근성 향상:** 상위 알고리즘들을 오픈소스화하고, Napari와 Docker 컨테이너에 통합하여 생물학자들이 코딩 지식 없이도 쉽게 사용할 수 있도록 접근성을 크게 개선했습니다.

## 📎 Related Works

* **특정 모달리티 세포 분할:** 형광 및 질량 분석 이미지 (Jackson et al., Lee et al.), 혈소판 DIC 이미지 (Kempster et al.), 박테리아 이미지 (Cutler et al.), 효모 이미지 (Bunk et al., Dietler et al.).
* **일반화된 세포 분할 알고리즘:** Cellpose (Stringer et al.), Omnipose (Cutler et al.), Cellpose 2.0 (Pachitariu and Stringer).
* **생의학 이미지 데이터 과학 대회:** Cell Tracking Challenge (CTC, Ulman et al., Maˇska et al.), Data Science Bowl (DSB, Caicedo et al.), Colon Nuclei Identification and Counting Challenge (CoNIC, Graham et al.).
* **CNN 기반 아키텍처:** U-Net (Falk et al.), DeepLab (Chen et al.).
* **Transformer 기반 아키텍처:** SegFormer (Xie et al.), Multiscale Attention Network (Fan et al.).

## 🛠️ Methodology

1. **챌린지 설계:**
    * NeurIPS에서 국제 챌린지를 조직하여 범용적이고 효율적인 세포 분할 알고리즘 개발을 목표로 했습니다.
    * **개발 단계:** 1,000개의 라벨링된 이미지와 1,725개의 비라벨링된 이미지를 제공하여 알고리즘을 개발하고, 101개의 튜닝 세트에서 성능을 평가하며 순위표에 점수를 공개했습니다.
    * **테스트 단계:** 상위 30개 팀은 알고리즘을 Docker 컨테이너로 제출하여, 완전히 숨겨진 422개의 홀드아웃 이미지(미학습 데이터)와 2개의 전체 슬라이드 이미지(WSI)로 구성된 테스트 세트에서 평가받았습니다.
    * **평가 지표:** 정확도는 F1 점수($F1 = \frac{2 \times \text{Precision} \times \text{recall}}{\text{precision} + \text{recall}}$)로, 효율성은 이미지당 실행 시간으로 측정했으며, "순위-후-집계(rank-then-aggregate)" 방식을 사용하여 최종 순위를 결정했습니다.
2. **데이터셋 큐레이션:**
    * 20개 이상의 생물학 연구소와 50개 이상의 다양한 생물학적 실험에서 밝은시야(brightfield), 형광(fluorescent), 위상차(phase-contrast, PC), 미분간섭대비(differential interference contrast, DIC)의 네 가지 모달리티에 걸친 현미경 이미지를 수집했습니다.
    * 세포 기원, 염색 방법, 현미경 유형, 세포 형태의 다양성을 확보하여 일반화된 모델 학습을 촉진했습니다.
    * 훈련 세트에는 1,000개의 라벨링된 이미지가, 테스트 세트에는 422개의 이미지가 포함되었으며, 테스트 이미지는 모두 새로운 생물학적 실험에서 확보되어 알고리즘의 일반화 능력을 평가했습니다.
3. **최우수 알고리즘 (T1-osilab by Lee et al.):**
    * **모델 아키텍처:** SegFormer [48]를 인코더로, Multiscale Attention Network (MA-Net) [13]를 디코더로 사용하는 Transformer 기반의 인코더-디코더 프레임워크를 채택했습니다. Cellpose [41]에서 제안된 바와 같이, 셀 확률 맵과 수직/수평 기울기 흐름을 공동으로 예측했습니다.
    * **학습 전략:** 공개된 현미경 이미지 데이터셋으로 사전 학습한 후, 챌린지 데이터셋으로 미세 조정하는 2단계 학습 프로세스를 사용했습니다.
    * **데이터 증강:** 셀별 강도 교란 및 셀 경계 제외와 같은 셀 인식(cell-aware) 증강 기법과 일반적인 강도/공간 증강을 결합했습니다. 희귀 모달리티는 비지도 클러스터링을 통해 과표집(oversample)하여 학습했습니다.
    * **지속 학습:** 미세 조정 시 **Cell Memory Replay** [5]를 사용하여 기존 지식의 재학습을 통해 **Catastrophic Forgetting**을 방지했습니다.
    * **추론:** 슬라이딩 윈도우 전략을 활용하여 대규모 이미지 처리의 효율성을 높였습니다.

## 📊 Results

* **최고 성능:** T1(osilab) 알고리즘은 89.7%의 중앙 F1 점수(IQR: 84.1-94.8%)를 달성하며 다른 알고리즘들을 명확한 차이로 능가했고, 특히 점수 분포에서 이상치 수가 적어 높은 견고성을 입증했습니다.
* **정확도 및 효율성:** 최상위 알고리즘들은 약 2초(1000x1000 이미지)의 추론 시간과 3GB 미만의 GPU 메모리 소비로 우수한 정확도와 효율성 균형을 보여주었습니다.
* **통계적 유의성 및 순위 안정성:** T1 알고리즘은 모든 다른 알고리즘에 비해 통계적으로 유의미하게 우수했습니다 ($p < 0.05$). 또한, 부트스트랩 샘플링과 다양한 순위 체계에서 일관되게 1위를 유지하여 높은 순위 안정성을 보였습니다.
* **기존 SOTA 방법 능가:** T1 알고리즘은 Cell Tracking Challenge의 최고 솔루션인 KIT-GE, Cellpose, Omnipose 및 이들의 변형 모델들보다 훨씬 높은 정확도를 기록했습니다. 특히 KIT-GE보다 49.9%, Cellpose-pretrain보다 24.4%, Omnipose-pretrain보다 58.9% 높은 F1 점수를 달성했습니다.
* **Catastrophic Forgetting 관찰:** 새로운 외부 테스트 세트에서 Cellpose 및 Omnipose의 파인튜닝 모델은 사전 학습 모델에 비해 성능 저하(Catastrophic Forgetting)를 보인 반면, T1은 Cell Memory Replay를 통해 이를 효과적으로 극복했습니다.
* **시각적 비교:** T1(osilab)은 다양한 현미경 유형, 세포 유형 및 이미지 대비에 걸쳐 탁월한 정확도를 보여주었습니다.

## 🧠 Insights & Discussion

* **Transformer의 우월성:** Transformer 기반 알고리즘은 CNN에 비해 전역적 문맥 파악 능력, 큰 모델 용량, 뛰어난 전이 학습 능력 덕분에 기존 SOTA 세포 분할 알고리즘보다 월등한 성능을 보였습니다.
* **데이터 다양성의 중요성:** 라벨링된 이미지와 라벨링되지 않은 이미지를 모두 포함하는 광범위하고 다양한 데이터셋이 일반화된 성능을 달성하는 데 결정적인 역할을 했습니다.
* **효과적인 전략:**
  * **다중 헤드 출력:** Instance segmentation을 거리 맵 회귀 및 셀 전경 의미론적 분할 태스크로 전환하는 방식이 기존 detection-then-segmentation 패러다임보다 우수했습니다.
  * **다양한 데이터 증강:** 전역 강도, 공간 증강 외에 셀별 무작위 교란, 모자이크 증강 등이 모델 일반화에 기여했습니다.
  * **효율적인 백본 네트워크:** SegFormer 및 ConvNeXt와 같은 효율적인 백본은 정확도-효율성 트레이드오프를 최적화했습니다.
  * **슬라이딩 윈도우 추론:** WSI와 같은 대규모 이미지 처리에 효율적인 전략임이 입증되었습니다.
* **비라벨링 데이터 활용의 미해결 과제:** 상위 팀들은 비라벨링 데이터를 활용하기 위해 일관성 정규화, 의사 라벨링, 불확실성 인식 메커니즘을 시도했지만, 분할 성능 향상에 뚜렷한 효과를 보이지 못하여 여전히 탐구할 부분이 남아있습니다.
* **Catastrophic Forgetting 및 해결책:** 전이 학습에서 흔히 발생하는 Catastrophic Forgetting (새로운 데이터에 맞춰 미세 조정한 모델이 기존에 학습한 지식을 잊는 현상)이 관찰되었으며, T1 알고리즘은 Cell Memory Replay를 통해 이 문제를 성공적으로 완화했습니다.
* **접근성 및 잠재적 응용:** 최상위 알고리즘들은 오픈소스화되고 Napari 및 Docker 컨테이너와 통합되어 생물학적 이미지 분석의 처리량을 가속화하고 정량적 생물학 연구에서 새로운 발견을 촉진할 잠재력이 있습니다. 응용 분야로는 암 미세 환경 분석, 다중 질량 세포 계측 이미징, 질병 진행 연구 등이 있습니다.
* **제한 사항 및 향후 방향:** 현재 챌린지는 2D 현미경 이미지에 국한되며, 사용자로부터의 상호작용 피드백 지원이 부족하고 분류 작업이 제외되었다는 한계가 있습니다. 향후 챌린지는 3D 이미지, 분류 작업, 그리고 **Biologist-in-the-loop** 시스템 개발을 포함하여 더욱 복잡한 문제를 다루어야 합니다.

## 📌 TL;DR

**문제:** 기존 세포 분할 방법은 특정 현미경 모달리티에 특화되어 있거나 수동 조정이 필요하며, 다양한 이미지 유형과 효율성을 모두 고려하는 범용적인 솔루션이 부재했습니다.

**방법:** NeurIPS에서 다중 모달리티 현미경 이미지 챌린지를 조직하여, 50개 이상의 다양한 생물학적 실험에서 수집된 1,500개 이상의 라벨링된 이미지로 구성된 대규모 데이터셋을 제공했습니다. 참가팀들은 Docker 컨테이너로 알고리즘을 제출하여 F1 점수(정확도)와 실행 시간(효율성)을 기준으로 평가받았고, 특히 미학습 데이터에 대한 일반화 능력을 중점적으로 측정했습니다.

**핵심 결과:** 최우수 Transformer 기반 알고리즘(T1-osilab)은 기존 SOTA 방법들을 크게 능가하며, 수동 개입 없이 다양한 현미경 이미지에 대해 견고하고 정확하며 효율적인 세포 분할 성능을 입증했습니다. 이 알고리즘은 Cell Memory Replay와 같은 전략을 통해 전이 학습의 Catastrophic Forgetting 문제를 극복했습니다. 또한, 상위 알고리즘들은 오픈소스화되고 Napari 및 Docker와 같은 사용자 친화적인 플랫폼에 통합되어 생물학 분야에서의 광범위한 활용 가능성을 열었습니다.
