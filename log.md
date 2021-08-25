+ Jobs (git commit b1c7491):
    + 969205: ndf=32, 200 epochs
    + 969206: ndf=48, 200 epochs -> too few wires hit, always the same. Mode collapse? Peaks in loss early on
    + 969207: ndf=8,   200 epochs
    + 972348: ndf=8,   800 epochs -> very stable, could go on? Activated wires too high
    + 970426: ndf=16, 800 epochs -> interesting change in loss after 400 epochs
    + 972342: ndf=32, 800 epochs -> edep and doca are not great...
    + 972306: ndf=16, 1600 epochs -> way off in distributions and activated wires

    + 994102: ndf=8, 1600 epochs
    + 994104: ndf=8, 2400 epochs
    + 994105: ndf=8, 3200 epochs
    + 994106: ndf=8, 4000 epochs
    + 1006814: ndf=16, 1600 epochs
    + 1006815: ndf=16, 2400 epochs
    + 1006816: ndf=16, 3200 epochs
    + 1006817: ndf=32, 1600 epochs
    + 1006818: ndf=32, 2400 epochs
    + 1006819: ndf=32, 3200 epochs
    + 
    + 1039872: ndf=32, 50 epochs
    + 1039860: ndf=16, 50 epochs
    + 
    + 1039994: ndf=32, 500 epochs + save every 10
    + 1039995: ndf=16, 500 epochs + save every 10
    + 1045479: ndf=16, networks2, 500e + s10
    + 1052098: ndf=16, networks3 (occupancy), 500e + s10
    + 1053746: ndf=16, networks4 (occ+bigger wire kernel), 500e + s10
    + 1056059: ndf=16, networks5 (just bigger wire kernel), 500e + s10
    + 1064892: ndf=12, n5, 500e + s50
    + 1064893: ndf=20, n5
    + 1064894: ndf=24, n5
    + 1: ndf=8, n5
    + 1066289: ndf=8, n5
    + 1066291: ndf=6, n5
    + 1066292: ndf=4, n5
    + +++ **networks6**: same kernel size over all wire-wise convolutions
    + 1066421: ndf=4, n6
    + 1066422: ndf=6, n6
    + 1066424: ndf=8, n6
    + 1068119: ndf=12, n6 -> goodish
    + 1066425: ndf=16, n6 -> unstable for some reason
    + +++ **chunk_size=24** (still net6)
    + 1068622: ndf=8, n6, 50e -> goodish
    + 1068149: ndf=12, n6 -> ~~unstable~~ recovered! goodish
    + 1068626: ndf=16, n6, 50e -> good. Doca colours a bit dark even though distribution doesn't indicate that
    + +++ **network 7**: two fewer conv layers in wire and p branch
    + 1072242: ndf=8, 100e
    + 1072243: ndf=12, 100e
    + 1072244: ndf=16, 100e -> seems worse than n6
    + +++ **network8**: two fewer conv layers in the common branch
    + 1072302: ndf=8, 100e
    + 1072304: ndf=12, 100e
    + 1072305: ndf=16, 100e
    + +++ **network9**: better residual connections
    + 1076519: 8
    + 1076520: 12
    + 1076521: 16, 200 -> not good in wire, not great in p…
    + +++ Revert back to number of layers in net6. Maybe try with better residual connections
    + +++ **network10**: branch of network6. kernel_size 513 in p branch.
    + 1097141: 12, 200e
    + +++ **network11**: kernel_size tapering
    + 1097156: 12, 200e
    + +++ **network12**: standard residual connections (+ gen dropout reduced 0.05 -> 0.02)kj
    + +++ **network13**: Residual Blocks (problem: ReLU before Tanh)
    + 1114102: 12, 200e
    + +++ **network14**: One more layer in p and w branch. Only one layer with kernel 513
    + 1118904: g12 d14, 200e
    + 1119431: g24 d28, 200e
    + +++ **network15**: Fix activation before generator Tanh / GumbelSoftmax
    + 1130654: g12 d14, 200e -> goodish distributions but too many wires hit, doca not great
    + 1130735: g24 d28, 200e -> similar to above, better doca (@120e)
    + +++ **network16**: Fixed LeakyReLU alpha to 0.2 in ResBlock(T)
    + 1156320: g12 d14, 200e -> good distributions, too many wires hit
    + +++ **network17**: Changed more conv to ResBlocks in Gen, one more ResBlocks in Disc, higher kernel_sizes in p and w, max kernel_size 1025.
    + 1160313: 12, 200e
    + +++ **train2**: occupancy regularisation
    + 1168514: 12, n17, 200e -> distribution ok, occupancy absolutely not good despite regularisation
    + 1168517: 12, n12, 200e -> Not great distributions and occupancy...
    + +++ **network18**: Only two ResBlocks. 1025-long convs are now in common branch.
    + 1176117: 12, occupancy reg (train2) -> activated wires suck
    + 1176119: 12, no occupancy reg (train) -> activated wires suck
    + 1178235: 12, old occupancy reg (train3) -> saaaaaaaaaaaame
    + +++ **train3**: old occupancy regularisation (variance of N(activated wires) for each sequence)
    + 1178261: 12, n12 -> meh
    + 1182224: 16, n12, batch_size=48
    + 1182223: 16, n17, batch_size=48
    + 1182222: 16, n18, batch_size=48 -> goodish @ 45e
    + +++ **new train3**: occupancy loss increased by factor 10
    + 1202249: 12, n18 -> not bueno at all
    + 1202250: 12, n12 -> meeeeeeeeeh

    + 1212368: train.py, n12, 800e -> very similar to below
    + 1212372: train4.py (embed noise), n12, 800e -> very nice actually… but net doesn't get relationship between wires and large patches are missing
    + +++ **network19**: back to having increasing kernel size in gen
    + 1221298: 12, 800e -> good, but occupancy loss plateaus after ~300 epochs
    + 1221995: 16, 300e -> occ losses reaches a minimum after 200, then goes back up?
    + 1221996: 64, 100e
    + +++ **network20**: downsamples in disc
    + 1238345: 6, 200e -> hmkay, occupancy loss goes down slowly but steadily 
    + 1238346: 10, 200e -> goodish, same as 6
    + 1238347: 14, 200e -> good, occupancy loss goes down a bit more rapidly
    + 1279058: 16, 400e -> good, same. Seems like nearby wires aren't necessarily recognised as being close by the nets...
    + 1279056: 20, 200e -> occupancy loss went down but still not great act. wires
    + 
    + +++ **network21**: smaller kernels
    + 1241020: 14, 800e -> distributions way worse as 24/32 below
    + 1241034: 24, 800e -> same v
    + 1241056: 32, 800e -> perfect distributions, bad activated wires
    + +++ **network22**: Causal temporal convolutions
    + 1271991: 16, 200e
    + 1273583: 16, 200e
    + +++ **23**:
    + 1273599: 16, 200e -> everything is off...
    + 1273601: 24, 200e -> no occupancy loss decrease, activated wires way out
    + +++ **24**: TC(1024) in both branches
    + 1279053: 16, 200e
    + +++ **25**: only one TC layer (1024) in common branch
    + 1279055: 16, 200e -> meh
    + +++ **26**: TC are now (2048)
    + 1279270: 16, 200e -> not great in distributions, occ loss constant
    + +++ **27**: kernels are 3 instead of 2
    + 1279271: 16, 200e
    + +++ **28**: kernel 3 + zero padding instead of replication padding
    + 1279272: 16, 200e. -> distributions better
    + +++ **29**: kernel 2 + zero padding instead of replication padding
    + 1279473: 16, 200e
    + +++ **30**: replace TCNTranspose in gen by normal TCN
    + 1293132: 16, 200e -> lol nope
    + +++ **31**: Downsamples in disc
    + 1293150: 16, 200e	 -> no change in occupancy
    + +++ **32**: try to balance disc and gen params with more disc layers (TCN+Res), spectral norm in all layers except final Linear
    + +++ **train5**: normal GAN loss -> so fast!
    + 1303904: 32, 800e -> doesn't learn
    + +++ **33**: spectral norm on final Linear of Disc.
    + 1303909: 32, 800e -> doesn't learn
    + +++ **train6**: WGAN without GP:
    + -> no bueno
    + +++ **34**: I forgot spectral norm in projection layers of TCN and Res. Fix various minor activation problems
    + 1308820: 32, 800e -> loss blows up around 400e. Bad occ. Distributions meh.
    + 1308821: 16, 800e -> loss explodes after ~130 epochs. Distributions still okay. Occupancy bad.
    + ~~1308822: 64, 200e~~
    + ndf=12, 100 epochs:
   + +++ **35**: TCN all have 1024 receptive field. Tried to balance gen and disc params. Gen dropout 0.05 -> 0.15
   + occ goes down!
   + 1310842: 16, 800e, GAN
   + 1310841: 32, 800e, GAN
   + 1310843: 16, 400e WGAN -> good distributions, bad occ, as usual
   + ~~1310844: 32, 400e, WGAN~~
   + 1310858: 32, 400e, WGAN -> good distributions, bad occ, as usual
   + +++ **36**: TCN are all 2048.
   + 1310872: 32, 800e, GAN -> blows up
   + 1310873: 32, 400e, WGAN -> good distributions, bad occ
   + +++ **train5 / train_wgan + net36**: 2048 receptive field. disc now takes in wire position. Generator output is multiplied with [wire_x, wire_y] matrix to comply with that.
   + 1326702: 16, 200e, WGAN -> Blew up completely.
   + 1326705: 16, 200e, GAN -> Blew up completely. But probably because of bad normalization.
   + +++ Fixed wire_to_xy normalization
   + 1337076: 16, 200e, GAN -> blows up regardless
   + 1337078: 16, 200e, WGAN -> Doesn't blow up! But all hits on the same 10 wires.
   + +++ **37**: TCN receptive field not 2048 everywhere. Tapers off in Gen and Disc (1024 -> 256)
   + 1337088: 16, 200e, GAN
   + 1337084: 16, 200e, WGAN -> All hits concentrate on few wires
   + 5: 32, 20e, WGAN -> good distributions, occ bad
   + 
   + +++ **39**: copy of networks11 (before residual connection shenanigans)
   + 1344343: 16, 200e, WGAN
   + 1344357: 32, 200e, WGAN -> __Interesting… It's not bad but it's not good either..__ All hits on inner wires.
   + +++ **40**: remove kernel_size=1 convs
   + 0: 6, 80e, WGAN
   + 1344542: 16, 200e
   + +++ **42**: Res Convs to up/down sample in Gen and Disc.
   + 1346494: 16
   + 1346497: 32
   + +++ **43**: linear interpolation in Disc, fix in ResBlock (again?) ~~Wire distribution looks better already.~~ Nah it's the same...
   + 1348191: 16 -> Good results but all hits concentrated on inside of CDC...
   + 1348192: 32 -> Same
   + 
   + +++ **45**: Remove downsamples in disc. Feature maps increase toward n_wires
   + 1353645: 16 -> Looks worse than 43. All hits on <10 wires.
   + 1353648: 32 -> Same
   + +++ **46**: One more Res layer in p and w branch, more kernels. Dilated convs in Disc common branch
   + 1353767: 16
   + +++ **47**: Feature maps decrease toward n_wires
   + 1353935: 16, 200e
   + 1353936: 8,   200e
   + 1353937: 32, 200e
   + 
   + +++ **48**: reintroduce TCNs to reduce N parameters
   + 1353953: 4, 200e -> too few modes
   + 1353955: 8
   + 1353956: 16
   + 1353957: 32
   + +++ **49**: No TCN, no ResBlock. No downsample in Disc. No kernel_size=1 conv layer.
   + 1361678: 16, 200e -> All hits on inner layers, very few wire diversity
   + 9: 8, 100e
   + +++ **50**: kernel_size=1 conv at the end of Gen only.
   + 1361950: 16, 200e -> speed up is massive from this. We probably NEED to have it.
   + +++ **51**: Minimal number of TNC layers, no k_s=1 layer in disc
   + 1363225: 16, 200e
   + +++ **52**: smaller k_s toward end/start of G/D
   + 1368441: 16 -> same as 24, bit worse distributions
   + 1368442: 24 -> distributions ok. hits concentrated on 10 wires
   + 
   + 1374536: 16, train_wgan2 (wire position interpolation between real and fake, 0.1 scale)
   + 1374558: 16, train_wgan2 (1.0 scale on w interpolation loss) 
       + -> Low diversity for both. See note at bottom.
       
   + +++ **53**: More convs at the end of G
   + 
   + +++ **54**: More convs at start of D
   + 1384059: 16
   + 1384152: 24
   + 1384272: 28 -> Goodish, but too often hits the same wires...
   + +++ **55**: From 43, add conv projections
   + 15: 8, noise_level=0.05, 
   + 1390944: 8, noise_level=0, same tau as 15 -> no noise seems slightly better
   + 1400003: faster tau decay, noise_level=0
   + 1400004: + recommended betas from 'Improved WGAN training'
   + 1400005: + weight decay in disc -> Blew up
   + +++ **56**: No spectral_norm in disc
   + 17: 8 -> Losses seem okay. Could go on
   + 1402276: 16, 200e
   + ## 1403426: 16,
   + 1403427: 16 (same as 26?)
   + 1403428: 16 (same as 26?)
   + +++ **57**: LayerNorm in disc
   + 18: 8 -> Not sure if I'm applying LayerNorm properly. Maybe we should permute the tensors and apply along channels ?
   + 1403433: 8, 100e
   + 1403434: 16, 200e
   + +++ **58**: (should-be) Better LayerNorm usage
   + 19: 8 -> ok
   + 1403483: 8, *800e*
   + 1403478: 16, 200e
   +  -> Not ok. Losses diverge badly. 
   + +++ **Bump** Try with wire_pos normalization to [-1, 1]
   + 20: 8, 100e -> looks quite unstable still
   + 21: 8, 20e, no weight decay ->
   + +++ **59**: InstanceNorm
   + 1435424: 16 -> Loss blows up repeatedly but results are almost okay
   + +++ **60**: No normalization in D
   + 1436035: 16, 200e
   + 1436037: 16 + AdamW -> Looking better than Adam (e60). Check at e200
   + +++ **61**: Replace BatchNorm in G by LayerNorm
   + 1436038: 16, 200e
   + 1436040: 16 + AdamW -> very little learning in both cases xx
   + +++ **62**: Replace BatchNorm in G by InstanceNorm
   + 1436902: 16, 200e -> No peaks in loss
   + +++ **63**: Replace G LeakyReLU by ReLU
   + Summary: 
       + AdamW, noise_level=0, betas=(0.0, 0.9), tau decay x0.99 / it, min tau=0.1
       + InstanceNorm in G, no norm in D, ReLU in G, no dropout, gumbel_softmax
   + 1436913: 16, 200e
   + 1438057: 16, 200e, tau lower limit 1e-4 -> 1e-1 ==> Results much better than 1e-4
       + 1446196: Continuation (+800e, total 1000e)
       + -> Wire diversity is still wayyyyyyyy too low.
   + +++ **64**: Normal softmax
   + 1447388: 16, 200e
   + 
   + +++ **train_wgan4**: gradient penalty tests
   + 23: always fake -> spikes like crazy
   + 24: always real -> better loss
   + +++ **65**: remove InstanceNorms that act just before residual connection in G
   + 25: big peak at the start…
   + 1470729: 16, 200e -> eventually distributions good, wire bad.
   + +++ **66**: InstanceNorm after adding skip connection to output of conv in G
   + 26
   + 1474702: 16, 200e, fake_w in gradient_pen half the time
   + 1474703: 16, 200e, always real_w in gradient_pen. Loss has no peak, so it's better?
   + +++ **67**: 65 + gumbel_softmax
   + 1474720: 16, 200e, gradient_pen 50% real_w -> 
   + 1474728: same, 100% real_w
   + +++ **68**: 66 + gumbel_softmax
   + 1474722: 16, 200e  gradient_pen 50% real_w
   + 1474729: same, 100% real_w
   + +++ **69**: 68 + restore InstanceNorm in ResBlockTranspose + add 2 convw layers in G
   + 1483261: 8, 800e -> @70e does not seem to improve wire diversity problem, still doesn't at 800.
   + +++ **70+train_wgan4**: GAN+AE architecture, but no AE regularisation yet ()
   + 1504925: 12, 200e
   + 29: 8, 20e, Adam
   + +++ **71 + train_wganae**: AE BCE loss, encoding_dim = 16, n_critic=1, concatenated GP gradients so we only compute once
   + 34: 4, 10e, Adam 1e-3
   + 35: 4, xe,   Adam 1e-4
   + 38: Adam, 1e-4, 1 epoch of only AE
   + 39: AdamW	, 1e-4
   + 40: No bias in last layer of disc
   + 41: No weight init, AdamW(1e-3) -> No weight init much better
   + 42: Yes weight init, AdamW(1e-3)
   + +++ **72 + train_wganae2**: no weight init, ae AdamW 1e-4, AE single-layer Conv1d
   + 1538239: n_critic=5
   + +++ **73**: one more layer in AE for generality
   + 44: 4, 20e
   + 1539432: 16, 200e
   + +++ **74**: replace ndf/ngf by encoded_dim in AE
   + 1539167: 16, 200e
   + 1539433: 12, 800e
   + +++ **train_wganae3**: Instead of applying BCE on sequences, flatten sequence with batch dimension so our tensor becomes a tensor of dimension (batch * seq_len, n_wires)
   + 1551322: 12, 800e
   + 1551323: 16, 200e
   + 1551324: 8, 200e
   + 45: n74, 4, 20e
   + 1582406: 8, 200e, CosineEmbeddingLoss -> not hitting wires close enough
   + +++ **75**: Replace residual blocks by TCN blocks in D
	 	    + [x] Replace python list by ModuleList in TCNBlock
   + 1554672: 8, 200e 
   + 1554665: 12, 200e
   + +++ **76**: Replace large-kernel convs by TCN in G
   + 1555004: 8, 200e
   + 1555005: 12, 200e
   + 1555008: 16, 200e -> Loss sometimes peaks, but results ok. Diversity is greater than with ResBlocks, for some reason...
   + -> AE loss plateaus at 10^-2.5 after ~100 epochs. Can we make it better?
   + 1577830: 8, 200e, CosineEmbeddingLoss
   + 1605293: 8, 800e, wganae4 -> 
   + +++ **77**: 76+Increase encoded_dim to 64
   + 1575555: 8, 200e, n_critic=5 -> Good results
   + 1576611: 8, 200e, n_critic=1 -> Diversity problem. G always hits the same wires.
   + 1577839: 8, 200e, n_critic=5, CosineEmbeddingLoss -> Recovered after blow up but hits everything
   + ~~1602711: 8, 200e, DistReg~~ -> Uses too much RAM!
   + +++ **78**: 76+One more layer in enc/dec. encoded_dim=16
       + 1575653: 8, 800e, n_critic=5 -> Diversity and distributions much better than n_critic=1
       + 1576604: 8, 200e, n_critic=1 -> Very low diversity
       + 1577847: CosineEmbeddingLoss -> Glitch at ~80e in GAN loss…
       + 1602721: 8, 200e, wganae5 DistReg (CosineEmbedding, dist x1)
       + 1688457: 8, 200e, wganae9 (CosineEmbedding, dist x0.2)
       + 1691234: 8, 200e, wganae10 (BCE AE loss, dist x0.2)
       + 1743200: 8, 200e, wganae11 (BCE, dist x0 NO DIST REG) -> No bueno. 
           + __===> Dist reg helps...?__
       + 1744943: 8, 200e, wganae12 (11+ AdamW everywhere, __default betas__) -> Explosion
           + __===> ~~Don't use AdamW everywhere.~~__ See 1766409
       + 1756708: 8, 200e, wganae13 (12 (default betas)+ BCE, dist x1) -> blows up, but slightly recovers. Distributions are not good, distance gets better, time_diff is good.
           + 
       + 1766409: 8, 200e, wganae15 (12 (default betas) + AdamW only in disc (not gen, not AE))
           + Worse than with dist reg
       + ~~1769154: 8, 200e, wganae16 (12+ AdamW in disc and AE (not gen)) Ah that's wganae10~~
       + 1769352: 8, 200e, wganae16 (12 (default betas)+ AdamW NOWHERE (only Adam) but default betas) -> __very bad__
       + 
       + 1769593: 8, 200e, wganae17 (10 + dist x1) -> __quite good__
       + === I reset the betas in wganae12.
       + === Start again from wganae17
       + 1781917: 8, 200e, wganae19 = 17 + AdamW in gen -> quite similar to 17
       + 1781965: 8, 200e, wganae20 = 17 + no dist reg -> __Clear difference in distance plot and event display__
       + 1783743: 8, 200e, wganae21 = 17 + betas 0, 0.9 in AE, AdamW everywhere -> Distance reduces faster than 19/20
       + 1784191: 8, 200e, wganae22 = 21 + betas 0.5, 0.9 AdamW everywhere -> difficult to compare with 21. Results/loss quite similar
       + 1803302: 8, 200e, wganae23 = 17 + betas 0.0, 0.9 in AE + x1 dist reg -> works, priority wires remain and hit distribution in each sample is asymmetric. Occupancy still high.
       + 1809099: 8, 200e, wganae24 = 23 + no dist reg -> dist does not change @160.
       + 
       + 
   + +++ **79**: 78+Residual connections in Enc/Dec
   + 1577115: 8, 200e, n_critic=1
   + 1577849: n_critic=5, CosineEmbeddingLoss
   + 79 and 76 have very similar performance with AE
   + 1602728: 8, 200e, DistReg
   + +++ **80**: 74 + reverse order of convw in G: 513 is at the end now
   + 1586065: 8, 200e, CEL -> ok, not great, too many wires
   + 1602735: 8, 200e, DistReg
   + +++ **81**: no downsampling in disc
   + 1638392: 8, 800e -> compared to 653 (n78), the distance looks better. Is that because of no downsampling, or because of the other changes made in n79, n80 since? Some wires are getting overwhelmingly activated tho...
   + +++ **train_wganae6**: distance to previous hit regularisation
   + 1645859: 8, 200e, n78
   + 1646081: 8, 200e, n79
   + 1646083: 8, 200e, n80
   + 1646085: 8, 200e, n81
   + +++ **train_wganae7**: Add disc.enc parameters to G optimizer. Save dist-var and dist-mean losses to save file.
   + 1651866: 8, 200e, n81. Didn't save var and mean loss. -> Hmmmm..… Goodish but few wires get hit all the time
   + 1653230: n80
   + 1653234: n79 -> Nah
   + 1653237: n78 -> Meeeeeeeh
   + ~~1656294: n77~~
   + +++ **82**: 78 + LeakyReLU + Tanh output for encoder
   + 1656504: 800e, no distance mean/var reg
   + 1656466: 800e distance mean/var reg
   + +++ **83**: 81+ 1 layer in enc/dec + LeakyReLU and Tanh in enc
   + 1657201: 8, 800e, no mean/var reg
   + +++ **84**: go back to simple convs (not ResBlocks, tcn) everywhere
   + ~~1669600: 8, 200e, dist+mean+var reg~~
   + +++ **train_wganae9**: train_wganae5 (make euclidean distance match real geom) + reduced distance loss x0.2
   + 
   + +++ **85**: Add difference with previous hit in sequence to G
   + 1671400: 8, 200e, wganae9 
   + +++ **86**: Add difference with previous hit in sequence to D
   + 1673436: 8, 400e, wganae9 (weaker dist reg in AE)
   + 1673439: 8, 200e, wganae5 (stronger dist reg in AE)
   + 1673442: 8, 200e, wganae4 (only AE loss in AE training)
   + +++ **87**: concatenate rather than add subtracted w
   + 1673449: 8, 200e, wganae9
   + +++ **88**: difference conv in common branch (G)
   + 0 (20e):
   + 1684114: 8, 200e, wganae9
   + +++ **89**: 88 + equivalent in D
   + 1684917: 8, 200e, wganae9 -> doesn't look too good @105e
   + 
   + +++ **train_wganae10**: wganae9 + BCEWithLogitsLoss instead of CosEmbeddingLoss
   + 50: 4, n78
   + 1691234: 8, 200e, n78 (downsampling D) -> less priority wires, better distance, worse activated
       + 1715882: Continued (+400e) -> actually interesting results. The distance gets quite good.
   + 1691351: 8, 200e, n81 (no downsampling D) -> good but priority wires
       + 1715893: Continued (+400e)
   + 1705122: 4, 350e, n78
   + +++ **90**: 78 + force_prev_wire
   + 1696896: 8, 200e, wganae10 -> Good distributions, better distance, too many activated
   + +++ **91**: 78 + one more AE layer
   + 52:
   + +++ **92**: 91 + residual downsampling in D
   + 53:
   + 
   + +++ **93**: 92 + residual upsampling in G
   + 54:
   + 1711519: 4, 200e, wganae10 -> Meh
   + +++ **94**: remove res down/up sampling, remove one AE layer
   + 55:
   + 1716168: 8, 400e -> distance gets better and distributions are okay. Wire diversity is bad though, priority wires.
   + +++ **train_wganae11**: train_wganae10 + no dist reg
   + +++ **train_wganae14 / n95**: wganae10 / n78 + move AE outside of G and D
   + 1771897: 8, 200e -> @170 Clean distributions, clean loss. Dist is bad.
   + 1785639: 8, 200e, t14.2 dist x0.2 -> x1
   + 1785647: 8, 200e, t14.3 dist x0.2 -> x5
   + 1785712: 8, 200e, t14.4=t14.2 + AdamW (betas=0.0, 0.9) for AE and G -> Looks like failure. Always hits the inner layers.
   + 1803940: 8, 200e, t14.5=t14.2 + decode and reencode output of G 
       + -> Can't find the same dist-reg effect...?
   + 
   + +++ **train_wganae14.6 / n96**: D takes in one_hot + 1e-7 -> 1e-8 in sqrt
   + Actually, bad idea. D should really take an encoded position. 
   + 1818576: 8, 200e -> doesn't learn dist
   + +++ **train_wganae25**: wganae14+ Add decoder params to G optim
   + 1820381: 8, 200e -> doesn't learn dist
   + +++ **train_wganae26**: wganae25+ Add encoder params to D optim
   + 1825648: 8, 200e -> no workerino
   + +++ **train_wganae27**: wganae26+ betas (0.0, 0.9) for AE
   + 1831481: 8, 200e -> no workerino
   + +++ **train_wganae28**: wganae27+ decode + gumbel + encode
   + 1831787: 8, 200e -> __FINALLY yes workerino__ but overfit, priority wires.
       + **So the working formula is: G learns dec, D learns enc, dist reg.**
   + +++ **train_wganae29**: wganae28+ normal softmax
   + 1831974: 8, 200e
   + +++ **n97**: wganae28 (magic formula) + Tanh in Enc, Sigmoid in Dec like medGAN
   + 1844153: 8, 200e -> __Loss looks much smoother I think__
   + +++ **n98**: n97 + Tanh output on encoder 
   + 1844170: 8, 200e -> Looks like it's starting to learn dist at the very end..?
   + +++ **train_wganae30**: 28+ AdamW replaces Adam
   + 1844798: 8, 200e, n95 -> Compare with 787. Loss looks similar. Priority wires but less so.
   + +++ **wganae31**: 30+ default betas
   + 1845702: 8, 200e, n95 -> Loss goes way higher than with betas=0.0, 0.9
   + +++ **train_wganae32**: 30+ betas (0.9, 0.99)
   + 1847188: 8, 200e, n95 -> Distributions bad, heavy wire priority.
   + +++ **n99**: new arch. No downsampling in disc. New Res class.
   + 1848008: 8, 200e -> distributions bad, distance bad, wire good
   + +++ **n100**: Tanh output to Enc and G's encoded wires. Don't activate D before output
   + 
   + +++ **n101**: 100+ fix my residual blocks…
   + 1853175: 8, 200e -> LEARNS! Dist as well!
   + +++ **wganae33**: 28+ no dist reg + tau * 0.999 + pre-train for 5 + option to change encoded_dim in args
   + +++ **n102**: Don't split the tensor in G, remove bias=False in AE,
   + 62
   + +++ **n103**: PReLU -> ReLU
   + 63 -> doesn't seem better…
   + +++ **n104**: Add activations in G when applying convu
   + 64 -> doesn't fix it either
   + +++ **n105**: 102+ Sigmoid in Dec, Tanh in Enc
   + 65
   + 66: with weight init in gen
   + +++ **n106 + wganae34**: 105 + replace conv1d in AE by Linear for speed + Res(4) in D
   + 0: can't really tell if its any faster…
   + 1861372: 8, 100e -> Bad results. Distributions bad, wire bad.
   + 1861402: 8, 100e, betas = (0.3, 0.99) [wganae35] -> distributions bad, dist bad, wire good
   + +++ **wganae36**: wganae34+ replace BCELoss by CrossEntropyLoss
   + 1866294: 8, 100e, no pretrain -> Not bad tbh, but distance not good.
   + 69 :^) enc_dim=4
   + 70: enc_dim=8
   + +++ **n107**: add conv layer to w and p branch in G
   + +++ **wganae37**: train only ae
   + +++ **n108**:
   + 1880093: 8, 200e, wganae37 -> Not very good
   + +++ **n110**: bring old net to new AE-pretrained stuff
   + 1880092: 8, 200e, wganae37 -> Better distributions
   + +++ **wganae38**: 37 + betas=(0, 0.9) and AdamW for D
   + 1883305: 8, 200e, n108 -> distributions not as good as 110
   + 1883309: 8, 200e, n110 -> Good distributions, but distance tends to be too high. Seems better than Adam
   + +++ **wganae39**: Add dist reg to AE pre-training
   + -> ae_states_v1
   + 1900859: 8, 100e, n110
   + +++ **wganae40**: Do not allow G/D to change Enc/Dec
   + 1900862: 8, 100e, n110
       + cont'd 1909124: 8, +200e, n110 04:30:52
   + 1907810: 8, 200e, n111 -> 03:53:59 (14039 seconds)
   + +++ **n111**: Simpler G: fewer parameters by reducing Linear output, Simpler D: 2 convs and 1 lin. Do not split features into branches
   + +++ **wganae41**: Don't AE the G output before giving to D
   + 1903451: 8, 200e, n111 -> Hard to tell between AE and no AE… 02:43:41 (9821 seconds): much faster
   + 1909998: 8, 200e, n110 -> Difference between this and with AE? (862)
   + +++ **n112**: one more branch conv
   + 1909980: 8, 200
   + +++ **wganae42**: 41+ betas (0.5, 0.99)
   + 1912676: 8, 200e, n110
   + +++
   + 79: n111, no gumbel
   + 81: n111, yes gumbel -> Looks the same to me
   + 80: 
       + enc_dim=2 final CELoss: 4.752271842956543
       + enc_dim=3 -------------> 0.55127804428339 (acc 96%, dist 0.4, norm 0.01)
       + enc_dim=8 -------------> 2e-5 (acc 100%, dist 0.001, norm 1e-5)
       + enc_dim=4 -------------> 0.01 (acc 99.9%, dist 0.3, norm 0.0009)
       + enc_dim=5 -------------> 0.003 (acc 100%, dist 0.06, norm 0.0001)
   + +++ **n114**: replace feature splitting by conv(ndf, n_features)
   + 1918890: 8, 200e, wganae42
   + +++ **n115**: crazy padded conv in disc
   + 82:
   + +++ **wganae43**: norm loss in AE pre-training -> ae_states_v2.pt
   + 1920143: 8, 200e, n115
   + 1920144: 8, 200e, n111
   + 1920145: 8, 200e, n110
   + +++ **n116**: more crazy convs in disc
   + 1930976: 8, 100e, n116, wganae43
   + 1933432: 8, 100e, n117, wganae43, no branch conv in G
   + 
   + 86: L1Loss for distance matrix
   + +++ **wganae44**: learn all wires at a time (see below)
   + 1933470: 8, 200e, n117
   + 1933471: 8, 200e, n116
   + 1933472: 8, 200e, n111
   + 1933473: 8, 200e, n110
   + 1939609: 8, 200e, n118 (more convolutions in G)
   + 1939787: 16, 200e, n119 (large kernel conv in G)
       + 1954855: cont'd +200e
   + 1950385: 16, 200e, n119
   + 1954862: 8, 200e,   n119
   + 
   + +++ **wganae45**: add noise to code in AE
   + -> ae_states_v5.pt (MSELoss)
   + -> ae_states_v6.pt (L1Loss)
   + +++ **wganae46**: use ae_states_v6
   + 1954735: n119, 8
   + 1954939: n119, 16
   + 1954755: n111, 8
   + 1954945: n111, 16
   + 1954763: n110, 8
   + 1954952: n110, 16
       + 1965283: cont'd +600e
   + 1964704: n119, 32
   + +++ **wganvae  /  networks120**: AE -> VAE
   + FIX VAE!
   + 94 -> ae_states_v7.pt
        cross entropy loss: 0.4269193181395531
        accuracy: 0.8666472637653351
        KL loss: 8.358506994247437
   + 1981363: 16, ???, n??
   + 1982426: 16, 2000e, n124
   + 1983142: 16, 2000e, n125 (more w-diff convs in D)
   + 106: add dist loss -> ae_states_v8.pt
   + 110: default Adam params for AE ->ae_states_v9.pt
   + 113: add enc parameters to G/D training
   + COmpare with 115 -> ok we need more parameters than ndf=4
   + 117 = 32
   + 116 = 16
   + enc params to in G/D optimizers
   + 118 = 16
   + 119 = 32
   + 120: 16, n131
   + 121: 16, n130, lr=5e-4, no enc params -> better
   + 122: 16, n130, lr=1e-3
   + 123: 16, n131, no enc params
   + 124: 16, n131, lr=1e-3
   + 125: 16, n132 (more convs G), lr=1e-3
   + 126: 16, n133 (more p convs D)
   + 127: 16, n134 fix G w-convs
   + 128: 64, n134 -> no bueno
   + 129: 16, n135 (more p convs G)
   + 130: 16, n136 (kernel size 1 in d) -> Weird, D loss stops going down here.
   + 131: 16, n137, w diff conv in D
   + 132: 16, n138, more w diff convs
   + 133: 16, n139, downsampling convs in D
   + 134: 16, n140: n130 + downsampling convs in D (compare to 129)
   + 135: repeat of 129, n135
   + 136: 16, n141 = n135 + downsampling convs in D
   + 137: 16, n142 = kernel_size 1 everywhere in G -> all hits have same energy
   + 138: 32, n141 -> loss spike
   + 139: 16, n143: kernel_size=3 everywhere in G except the end
   + 140: 16, n144: kernel_size=3 everywhere in G
   + 141: 16, n145: kernel_size=3 everywhere in D, more downsampling convs
   + 142: same but with chunk_stride=64
   + 143: 16, n146 = 2 more w-convs in G
   + 144: 12, ^
   + 2038601: 16, n144, 15000e
   + 145: 16, n147
   + 146: 16, n147, but 3 fewer downsampling convs
   + 148: re-add k_s=257
       + Not great...
   + 150: more feature maps
   + 2043027: 16, n147, 10000e.
   + 152: n148=n147+upsample linear
   + 
   + Try: Upsample mode -> meh
   + 2045311: 16, n150 (, 10ke -> loss spikes, bad
   + 157 -> spikes
   + compare 158 with 159 -> tanh vs relu
   + 
   + 160: n156 = ReLU in G (compare with 159 = PReLu)
   + n156 = 
   + n157: n156+ add convw_diff
   + 162: n157 = PReLU + convw_diff -> not good
   + 163: n158 = no convw_diff
   + 164: n159 = fewer maps in p branch G/D
   + 165: n160 = smaller kernel in p branch G/D (compare n159) -> no spike!
   + 166: n161 = n160 + fewer maps
   + 2050406: n162 = n161 + k_s=1 in end of G
   + 168: n163 = 	k_s=513 in D p-branch
   + 2050598: n163 -> too many act wires
   + 2053177: n167
   + 2053291: n168, 32
   + 2053438: n168, 16
   + 2053761: n169, 16
   + ae_states_v11.pt:
       + 2058285: n162
       + 2058297: n167
       + 2058202: n169
       + 
       + wgandist: feed in CDC wire distance to Discriminator
   + 188: wganvae n172 = sanity check before adding xy info to D
   + 189: wgandist n171
   + 191: n173 = concatenate xy info with p, w. ndf=3
   + 192: n174 = add xy info to p, w. ndf=3
   + 193: n174, ndf=8
   + 2101579: n174, ndf=16, 5000e -> hits same wire forever
   + 194: n175 = add xy only to w. ndf=8
   + 195: n176 = reduce feature maps in G ends
   + 196: ^ + ndf=32
   + 197: n177 = trying to make more sense of feature maps
   + 198: n177 @2000e pretty nice but priority wires
   + 199: n178 = one more conv in D common branch
   + __Compare jobs 198 and 199. Til now, G has k_s=1 in all final layers. D has a k_s=17 in entry layers only.__
   + 200: __n178__ = increase k_s in gen final layers to 17
   + 2107391: 16 -> one wire
   + 2107409: 8 -> one wire
   + 2107424: 24 -> one wire
   + 2107438: 32 -> one wire
   + 2107449: 48 -> one wire
   + 2107460: 64 -> one wire
   + Ah, I forgot to say --no-pretrain in the qsubs…
   + 2117405: 64
   + 2117406: 48
   + 2117412: 32
   + 2117435: 16
   + 2117441: 8 
       + They all end up picking 10 wires and hitting them over and over again
   + — Use soft gumbel softmax in training -> why not?
   + 2119398: 16 __compare with 2117435__
       + Better in terms of wires, but still pretty bad
   + 202: __n179__ = use xy difference instead of absolute
       + Doesn't seem as good at figuring out distance distribution...
   + 2119461: n179, 16
   + __n180__ = One more strided conv in all D branches
   + 203: 16
   + 204: Don't use difference in xy conv
   + 2119516: 16 -> __meh__
   + 2131525: 16 + Adam
   + Can also: tweak LR, AdamW, betas
   + __n181__ strided convs in G
   + 2120136: 16 -> __good__. Seems more diverse
   + 2131524: 16 + Adam -> still 
   + __n182__: dist convs in D, don't add xy but concatenate
   + 206: 16
   + 207: 16/8
   + 208: 16/8, chunk_stride / 4
   + __n183__: dont taper feature maps in G w-branch
   + 209: 16/8 -> looks like problem remains
   + 210: 16/8, LR=1e-5/3e-5
   + __Reduce LR to 1e-4 / 5e-4__
   + __n185__: k_s 513 in G
   + 211: 16/8
   + n187 = Tanh in G
   + 212 = n186 (k_s taper in G tips)
   + 214 = ^ 8/4
   + 213 = n187 (tanh)
   + 215 = ^ 8/4
   + n188 = ReLU
   + 216 -> explosion??
   + n189 = k_s=3 instead of 1
   + 218 -> same lr for D and G
   + 2138018: n189
   + 2138019: n188
   + 2138020: n187 -> best distributions
   + 219 -> 218 with ndf 64/32 
   + 220: n190 = simplify G upsamplings
   + 221: 220 + latent_dims=128
       + ReLU really messes with the generator, it seems… Or maybe it's something else
   + n191 -> Tanh G
   + 222
   + n192 -> 
   + I FORGOT NO-PRETRAIN AGAIN. only in 223 tho
   + 224 vs 223: interpolates_xy is always real
   + 226: interpolates_xy is always fake
   + 227: interpolates_xy is actually decoded interpolates_enc_w
+ Let's check that we can make it work without xy information in D
   + 228: n193 = fewer layers in G
   + n198 = back to n189 but we take out the xy term
   + 2168769: n198 (complex) -> Nope
   + 2168780: n197 (minimal) -> Bad too! Because of altered dataset? Probably doesn't help
+ **ALTERED DATASET**. This one I'm okay with (dataset_altered.py)
+ 239: n200 = minimal with ReLU activation in G and Dec, LeakyReLU(0.2) in D and Enc
    + Is ReLU bad? Maybe not.
    + 2169706: n202 = n200 + LeakyReLU in G
        + 243 cont'd
    + 2169926: n203 = n200 + Tanh in G
        + 242 cont'd
+ 2169548: n200, 16, 20ke
+ n201 = one more conv in G
+ n204 = upsample -> strided conv
    + j244	
+ n206 = k_s=1 everywhere
    + Bad results despite that
+ n207 = separate branches in disc
    + 248 -> first hints of good results (16)
    + 2172659: 16, 20ke -> struggles with wire
    + 2172660: 32 -> best. Can see good correlations already at 1000e. But it falls off afterward as the wire distribution get bad.
    + 2172661: 64 -> Very good correlations, except for wire. wire distribution still meh
+ n208 = two k_s=33 conv in D/G
    + 2172662: 16 -> nope
    + 2172663: 32 -> not really
    + 2172664: 64 -> nope
+ n209 = k_s=33 only in D
+ 2177303 -> doesn't seem to work
+ n210 = one branch conv in G, k_s=1 everywhere
+ 2178006 -> wire collapse. w-p correlation not learned
+ train_wgandist_hardGS = hard gumbel softmax
+ 2180247: n210, 32, 5000e -> @1200 best results so far. feature/wire correlations good but can do better.
    + Performance falls off ??? @5000e, wire has collapsed, distributions are meh
    + 2189950: Repeat with Adam. Compare @1200 and @5000
+ train_wgandist_noGS = no gumbel softmax when going from G to D
+ 2180885: n210, 32, 5000e -> @800 G seems to have no idea where to place hits
    + @5000, completely uniform wire distribution. __Clearly we need some form of wire sampling (GS) between G and D. This is probably so that they can match their representations of the categorical feature.__
    + 2193764: Repeat with Adam -> @400 same. @5k completely uniform, yas.
+ n211: balance G and D entry conv n features to be ndf
+ 2181510: 32, 5000e __soft GS__ -> @1800 struggling with wire distribution. @5k good but wire not good enough.
    + 2193912: Repeat with Adam -> Similar/same results as AdamW
+ 2181734: 32, 5000e, hard GS -> @500 much better wire distribution, wire / feature correlation is looking good too. @5000e heavy wire collapse, distributions bad.
    + 2193911: Repeat with Adam -> @4000 heavy wire collapse, same as AdamW
+ 2183779: 64, 5000e, hard GS -> @3500 looking like wire explosion. Yup. No bueno wires.
+ 254: 32, 400e, hard GS, enc_dim=16
+ 255: 32, 400e, hard GS, enc_dim=32 -> explosion
+ 257: 32, 400e, hard GS, enc_dim=32, Adam
+ n212 = one more hidden layer in AE
+ 256: 32, hard GS, enc_dim=4,
    + 2189940 cont'd, 2000e
+ 258: 32, hard GS, enc_dim=4, Adam
+ 2189946: hard GS, enc_dim=8, 2000e -> not bad but collapses on most frequent wires
+ [__AdaBelief__]:
+ 259: 32, soft GS, n211
    + 2194663: contd
+ 2194387: 32, hard GS, n211
+ 2194666: 32, hard GS, n212
+ n213: Sigmoid in Dec, Tanh in Enc, like medGAN
+ 2194672: 32, hard GS, 5000e -> no bueno @2000 -> no bueno in wires
+ 260: 400e -> struggling with wire distribution / correlations
+ n214: one more w conv (G)
+ 2195687: 32, 2000e -> no correlations
+ n215: LeakyReLU in enc and dec
+ 2196024: 32, 2000e  -> poor p-w correlation
+ 261: 
+ n216: hidden_size x 4 in first layer enc/dec
+ 2196096: 32, 2000e -> no p-w correlation
+ n217: no upsampling (G)
+ 262: 400e -> veeeeeeery good distributions, except for wire
+ 2202785: 2000e -> good dists, W is completely uniform
+ n218: one more branch conv D
+ 262: 100e
+ 263: 50e
+ n219: reduce AE to one layer (zero hidden layers)
+ 264: 400e
+ 265: 400e, faster tau decay -> we see wire collapse much faster 
+ 266: pretrain a little bit
+ 267: normal softmax (AdaB_softmax)
+ n220 = no ups, one layer AE
+ 2209786: 16, 1000e
+ 2209799: 8
+ 2209815: 64
+ 2209826: 32
+ 2209837: 32, enc_dim=16
+ n221 = (1) only critic on wire feature -> (2) 2 more G conv -> (3) 2 more D conv
+ 1. 269
+ 2. 271
+ 3. 4 more layers in AE
+ 272: make G learn enc
+ pretrain AE -> makes it worse
+ __274, n222__: enc_dim 128 allows G to generate the correct wire distribution
+ n223 = same with Tanh in Enc / w
+ __275, n223__: works same with tanh. Let's keep it
+ n224 = what if we increase the ngf before output instead
+ 276: enc_dim 4, goes to ngf x 16 before enc_dim
+ __NOPE__: __we needed waaaaaay larger enc_dim all along__
+ Let's see from what enc_dim onward we can get good results
+ _n223_:
+ 282: 96: okayish
+ 277: 64: not as good
+ 278: 32: nyeh
+ 279: 128: 
+ __n225__: bring back the features
+ 280: 
+ 2216371: 16, enc_dim=128, 2000e -> fall to the abyss followed by wire collapse
+ 2216382: 64, enc_dim=128, 1000e -> quite good but E-w correlation isn't perfect. Somewhat wire collapse. Looks like we need ndf quite high, but wire collapse still consistently happens. We need to get rid of it.
+ n226: reduce feature maps in D
+ 281: @200e wire collapse...?
+ _n227_: no branch conv in G
+ 2216398: 16, enc_dim=128 -> absolute wire collapse
+ _n228_: no branch conv in D
+ 2216402: 16 -> mnyeeeeeeeh. Why do we always end up hitting few wires more frequently than others???
+ 2216403: 32 -> meh. Not that different to 16 actually. Can't tell if better or worse...
+ _n229_: strided conv in G
+ 283:
+ 2216426: 32, 1000e
+ 2216427: 64, 1000e
+ 2216428: 128, 1000e -> not learning correlations (even p). Still slight wire collapse, but overall distributions is goodish.
+ 2216429: 256, 1000e
+ 2216430: 64, enc_dim=256, 1000e -> pretty much the same all around
+ 297 = 295 + faster tau decay
+ Am I decaying tau too fast? should I make it constant to stabilise the nets?
+ 298: tau=2, Adam
+ 299: tau=2, AdaB, n238
    + __Seems more stable to me__
+ 300: n240 = increase feature maps in D
+ 301 = 299 + rectify=True
+ 303: aeloss, n240 -> _is this bueno?_ Hm, wire collapse still happens at large epochs.
+ 2225449: n238, rectify. -> Despite constant tau, wire collapse occurs.
+ 2225450: n238, aeloss -> wire collapse, no correlation to be seen
+ 2225456: n240, aeloss -> better wire correlation __Difference = increasing feature maps in Disc__
+ n242 = one branch in D, with k_s=3
+ 305: 
+ 2229345: 32, enc_dim=64, n242, rectify -> similar to the one with aeloss. Good but w/E correlation lacking.
+ 306: ^
+ 2229346 ^ + aeloss -> looking alright after getting past some point. w/E correlation is lacking though.
+ 307: latent_dims=16
+ 308: latent_dims=32, ae_loss x 100
+ 310: latent_dims=512, rectify
+ 311: latent_dims=32, ndf=256 (13M params in D xd), n240 -> blew up
+ 312: latent_dims=32, ndf=128, n242, Adam -> slow-learning, 
+ Try: high latent_dims, no ae loss
+ Try: more chunk_stride?
+ 313: latent_dims=512, Adam -> meh?
+ 314: latent_dims=32, ndf=8 -> not doing too good with distributions. Looking like strings, hitting weird places.
+ 315: n243 = feature map taper in G -> Looks ok @100
+ 316: n243, chunk_stride x 16 -> hint of learning the w/E correlation?
+ 317: n244, fixed feature maps, latent_dims 32, enc_dim 64
+ n245: no bias in AE
+ 318: n245, same as 317
+ n246: same with bias=True in G
+ 319: meh
+ n247: bias=True everywhere
+ 320: n247
+ What seems to work somehow: 
    + n242 = single-branch discriminator, although w/E correlation is lacking
    + enc_dim=64, latent_dim=256, ndf=32.
    + Two branches in G, two convs layers in each.
    + Three up/down sampling convs in G/D.
    + n243 = same with feature-map tapering in G
+ n248: n243 + one more layer in branch convs G
+ 321: Looks like 320 and 321 are missing some weights (ndf+)
+ 322: 321 with ndf=64
+ n249 = 247 + one more layer in branch convs G
+ 323: ndf=64. Wait, ndf doesn't make a difference for 247
+ n250 = 247 + activation in D first layer
+ 324
+ 
+ 2241788: n247, 1000e, ndf=64, enc_dim=64, latent_dims=32 -> no bueno
+ 2241789: n248
+ 2241790: n249
+ 2241791: n250 -> only one with acceptable results. I have no clue what's going on
+ n251: increase feature maps in G/D to 512 at tips
+ 325 -> absolute hell @25e
+ n252: no 1024 -> 512 conv, 1024 everywhere instead in D (12M params)
+ 328: Explosion (no weight init)
+ 329: Weight init for D, G -> no explosion
+ n253 push the number of weights to the limit
+ 330: no weight init, One more instance norm in G
+ __doesn't work__.
+ __So when we get an early explosion, we should initialise conv weights__
+ Dist or Gen or both?
+ 331: only disc -> nope
+ 332: only gen -> nope
+ 335: gen + ae -> yas
+ 336: disc + ae -> nope
+ 333: gen + disc -> __required.__
+ 337: gen + disc + ae -> yas
+ n253: one more before projection InstanceNorm
+ 334: nope
+ 339: ok!
+ n254: no instancenorm
+ 2254960: 2000e, latent_dim=64, enc_dim=64, n254, Adam
+ 340: AdaB
+ n255: taper 512
+ 341: AdaB. Well, at least it's much faster.
+ 2254961: 2000e, latent_dim=64, enc_dim=64, n255, AdaB
    + Dists aren't good, wire isn't good.
+ 
+ 344: n256
+ 345: n255
+ 346: n257
+ 347: n258 = 3 branch convs in D
+ 348: ^ + more epochs
+ 349: n259 = 3 branch convs in G as well as D
+ 350: enc_dim=4
+ n260: residual connections in both G and D
+ 2276189: cont'd, +3000e, enc_dim=4, latent=64
+ n261: more layers
+ 2276480: 4000e, enc_dim=4
+ 2328702: 1000e, HeInit. Distributions good but p-w not good.
+ n262: no branches
+ 2290683: 4000e, enc_dim=4, latent=64
    + Stops learning wire distribution efficiently...
+ 354: HeInit
+ 355: HeInit + eps=1e-16
+ 2294199: HeInit + eps=1e-16, 4000e, enc_dim=4, latent=64 -> looks meh @ 2700
+ 2295376: same with eps=1e-12
    + He doesn't look great. Or is it n262? Try n261 with He
+ 2298900: n261, He
    + Big meh for He
    + Let's try He with mode='fan_in'
    + Seems the same.
    + Tried with Adam optimizer as well. Seems to not work at all.
    + We'll stick to N(0.0, 0.02) initialisation I guess...
+ 2304161: n261, He, fan_in
    + Exploded, but distributions are looking good! w-p correlation bad
+ In the meantime, experiment with net and normal initialisation
+ n263 = one more lin in D
+ 2308180: n263, normal init
    + Looks collapsed
+ 2309106: same, with rectify=True
    + Better but not good either
+ n264 = GBlocks = residual upsampling blocks in G
+ 2308804: n264, normal init
    + Pretty good. p-w corr. is good-ish
+ 2309105: same, with rectify=True
    + 
+ n265 = 8 GBlocks.
+ 2309006: n265, normal init
+ 2309107: same, with rectify=True
    + Distributions aren't exceptional. p-w correlation not found.
+ n266 = DBlocks = residual downsampling blocks in D
+ 2315475: enc_dim=4, ld=64, 1000e, n266 (eight g/dblocks, only one linear layer in D)
    + HeInit = kaiming for D, N(0, 0.02) for G
    + One huge loss peak, distributions look collapsed, wire is uniform
+ 2316199: same, n267 (two linears in D, up to 1024 feature maps in G/D), 1000e
    + Absolute explosion
+ Increased beta1 from (0.5, 0.999) to (0.8, 0.999) -> looks like the explosion is slightly less pronounced…
+ 374 -> lr reduced to 6e-5
    + Not much better
+ 376 -> lr=2e-4, betas=(0.0, 0.9)
+ 377 -> AdamHe
    + Pretty high peaks (12,000)
+ 378 -> AdamHe + betas=(0.0, 0.9)
    + Higher peak (1e-15)
+ 379 -> AdamHe, lr=2e-5
    + Peak at 1e8 -> even worse?? than lr=2e-4??
+ 380 = only 4 D/GBlocks, AdamHe
+ 381 = lambda_gp=50
    + not much better
+ 382 = n_critic=8
    + Somewhat Better
+ 383 = RMSHe, n269, n_critic=4
+ 384 = HeInit, n268, n_critic=8
    + Explosion 1e29
+ 2331639: RMS, 269
+ 2331640: RMS, 268
+ 2331641: HeInit, 269
    + Hints of better training…
        + Still explodes at large epoch
+ 2331642: HeInit, 268
    + No bueno
+ 385: HeInit, n270 = one fewer layer in G/DBlock
+ 386: n271 = Just a downsampling conv in DBlocks
    + Seems to work better
+ 2334063: HeInit, 1000e, n271, n_critic=8
    + __Features and correlations very nice @500 and @1000. Wire collapse though.__
+ 387: n272 = Two convs in DBlocks, no branch in D
+ 388: n273 = n271 + 4 more layers
    + Struggle is real.…
+ 389: same, n_critic=16
+ 392: n275, HeInit
+ 393: ^, AdamHe -> looks good actually.…
+ 394: n276, HeInit
+ 395: n276, AdamHe
+ 396: n277 = 3 blocks in either branch
+ 397: ^, AdamHe
+ 2338180: n277, 250e, AdamHe
    + Wire is uniform @250
    + 2352410: same, 4000e, chunk_stride=2048
    + Wire is still uniform
+ 398: n278 = add conv(3) to DBlock
+ 399: ^, AdamHe
+ 400: ^, HeInit, n279 = dropout(0.5) in D
    + Explosion
+ 2339947: AdamHe, n279, 300e
+ 401: n280 = p + w in D, 512 max feature maps
+ 2341706: n280, 300e
+ n281 = reduce by 2 common branch convs, reduce DBlock to single strided conv
    + 402: nice features, nice loss @25
    + Not learning wire...
+ n282 = D lin1 512 neurons
    + 403: nice features, nice loss @25	
    + Not learning wire… What did I forget?
+ 404: n271 to compare
+ 405: n264, to compare (properly this time)
    + Why no learn wire???????
+ 407: enc_dim=8
+ 408: n_critic=1
+ __n282 architecture with AdamHe is able to quickly and reliably learn the feature distributions, but the wire distribution is either learned very very slowly, or not at all.__
+ 2352429: n282, 4000e
    + Clearly not learning wire distribution or correlations, even after such long training
+ 
+ __DEBUG TIME__
+ 409: n283 = no downsample in D
+ __Downsampling in D has no effect__
+ 410: n284 = two more w convs in G
+ __AE initialisation has no effect.__
+ 
+ 413: n286
+ n287: no upsample in G
    + 414
+ __Upsampling in G has no effect__
+ n288: Add more residual connections in both nets
+ 2354139: 4000e
    + Nope
+ n289: remove all residual connections
+ 2354121: 4000e
+ 431 = no init AE
+ Add AE loss back to GAN training
+ 2365795: n282, 4000e, AEopt.py
    + Good but wire distribution trains quite slowly...
    + 2371510: _cont'd, 40,000e total
    + 2394012: _cont'd, 80,000e
        + **Affected by LR mistake**
        + +++ w distribution learned to perfection.
        + ––– Has trouble learning E-w relationship, even after 80,000 epochs (!)
    + **MISTAKE**: AE optimizer had lr=2e-4, whereas AEoptAE has 2e-3
        + Fixed in _train_wgandist_AdamHeAEoptLR.py_
        + 2398047: 40,000e, n282 with LR fix
            + Doesn't seem to learn w distribution at all...
+ 2366931: n282, 4000e, AEoptAE (higher AE lr, D/G don't change AE **independent AE**)
    + 2394059: _cont'd 80,000e_
        + struggling with W @20k
        + Doesn't learn w @60k e
+ 2369254: n278, 10500e, AEoptAE
    + Nasty peak in loss, why? Net architecture?
    + Looks like a collapse
    + Recovers after a while
    + 2391555: _cont'd, 40,000e_
        + Not stable. Canceled @16k
+ 2371433: n282, 40000e, AEoptAE_normaldata
    + Training looks stable, events are okay but sequence isn't good enough
    + 2393758: _continued, 80,000e_
        + Although feature **and** wire distributions are realised to perfection, the CDC is too busy and the distance between consecutive hits shows no sign of going down.
        + Loss is extremely stable though.
+ train_wgandist_AdaBHeAEoptAE.py = swap in AdaBelief
+ 444: Okay, looks fine but spikey, let's try it anyway
+ 2387050: n282, AdaBelief swapin -> unstable @8000
    + Unstable throughout. Eventually learns but w-E correlation isn't there. Canceled @30k
+ 
+ 2388211: n282, pre-trained AE, AdamHeAEoptAE
    + Okay-ish wire distribution. E-w corr. not visible.
+ 
+ train_wgandist_AdamHeAEoptAE_test.py: try with lower tau. What happens then?
    + Compare with 441 @500
+ 446: pretrain 10,000, AdamHeAEopt
+ __n294__: n282 + hidden layer in AE
+ 447: n294 add layer to autoenc. cross_entropy is lower after 10k epochs
+ 2399542: n294, 10k epochs pre-training, 40,000e, n294, dependent AE (AEoptLR)
    + Does not learn w distribution @9800
    + **Doesn't suffer from LR mistake**
    + Explosion -> wire collapse
+ 448: n294 added layer, but G and D don't change AE (AEoptAE.py)
+ 
+ 452: normal dataset, n294, pretrain 10k
+ 
+ Pretrain with CrossEntropyLoss, but then stop using AE optimizer and let the GAN modify the AE
    + Start from AEoptLR
    + _train_wgandist_AdamHeAEoptLROnlyPretrain.py_
    + 453: n294
    + 2402752: 40,000e
        + -> @1800 not learning the wire distribution/corr.
        + Explosion -> wire collapse
+ __AE loss goes wayyyyyy up by itself if we don't keep the AE gradient descent in the GAN loop.__
+ 
+ n295 = no down conv in D before common branch
    + 455: AdamHeAEoptLR
    + 456: AdamHe (no AE gradient descent in GAN loop)
        + No sign of learning wire. __We really do need to regularise the AE code somehow__
    + 2412150: no pretraining, n295, 40,000e, AEoptLR
        + Absolute wire collapse. Ur cancelled.
    + 2416292: no pretraining, n295, 40,000e, AEoptAE
        + Nah
+ n296 = non-downsampling residual blocks in D, non-upsampling convs in G's w branch
    + 2418638: pretrain 2k, n296, 20,000e, AEoptLR
        + Seems slow to learn w @1400e
        + Absolute wire collapse. No bueno
+ n297 = no branches in D
    + 459: AEoptLR
    + 460: AEoptAE
+ 462: 294, pretrained AE, different LR for pretraining and GAN training
+ 463: same, enc_dim=2, 2.5ke pretraining
+ 464: same, 10ke pretraining. Can't seem to get low-enough loss with this
+ 
+ 473: n294, pretrain with code loss for 6000 epochs
+ 2433797: n294, fixed AE at GAN training time (pretrain_codeloss_fixae.py)
    + Doesn't learn w
+ 
+ n298 = let D train its own independent wire encoder (w branch takes in n_wires)
+ + train_wganpretrain_codeloss_clip.py = use weight clipping and pass raw wire to D
+ 
+ 475=474 + soft gumbel
+ 476 = 475 + pretrain for 6ke (clip=0.01)
+ 477 = 476 + clip=0.1
+ 478 = 476 + clip=0.001
+ 
+ 480: n294, codeloss_code (D uses generated wire code directly, no gumbel involved)
+ 481: Generator adversarially trains the encoder to fool the discriminator
    + AE pretraining
+ 482: 481 + no code loss, no pretraining
+ 
+ 2441612: 10,000e, codeloss_code_genenc, no pretrain, n294
+ n299 = don't branch G until tip
+ 2441611: 10,000e, codeloss_code_genenc, no pretrain, n299
    + Overfits wire and features?
+ n300 = don't branch D after tip
+ 483: 
+ 2442567: 10,000e
    + Overfits wire and features?
+ n301 = reduce N layers everywhere. Try to make correlations show up somehow.
+ 2443311
+ n302 = all kernel sizes=1 everywhere
+ 485: 5000e -> __EVALUATE!__
+ Increase AE LR by x10
+ train_wganpretrain_codeloss_code_genenc_upLR.py
+ 487: 500e
+ 2443949: 20,000e, n294
    + 
+ 2443972: 20,000e, n301
+ Why doesn't the net understand that some wires get more energy than others?
    + BatchNorm
+ 495: RMSprop, wire weight
+ 496: Adam, wire weight
+ 2484044: Adam, wire weight, pretrained for 15000e	, 10k epochs
    + Distributions, correlations poor.
+ 502: same, normal data, pretrain for 10,000
+ 2486735: Adam, wire weight, normal data
+ 
+ n305 = n294 + bias=True
    + Replace with linear to speed up training
    + __Linear is much much MUCH faster when pretraining, we should keep it.__
    + 504: okay
    + 506: optimizer_ae LR reduced to 2e-4 like G/D
+ n306 = no branches
    + 505:
+ 507 = 50k pretraining epochs, n305 (Linear), altered data, AE LR 2e-4, code loss on
+ 509 = 50k more
+ __Other idea: Autoencode  the x-y position of wires?__
+ Wait code loss was enabled in pretraining…
+ 505: disable code loss
    + Can only reach ~20% acc after 10k its. Run for more?
    + Eval script consistently reports higher cross entropy loss than pretraining.
+ train_wgpt1 = fix AE pretraining code…
    + Acc can go up to 60% after 10k epochs
    + Eval script says 70%. Okay.
    + And agreement between pretraining AE loss and eval script AE loss (no GAN training yet)
        + Let's check that after some GAN training, the two still agree.
        + They don't. AE loss is now 35 according to eval....
    + 508: n306

    + 510: n305, 20k pretraining… Why is AE accuracy so much lower than 508 after 10k??
    + 511: same as 510 on different terminal		
        + ??? Why is accuracy so low???
        + ??? It goes back up if I change to network 306???????
        + Why is accuracy so much higher for the same cross entrpy????
        + __Weight initialisation seems to be causing the AE to not train__
            + Disabled weight init for AE
            + Not sure why it depends on G/D architecture though, probably a random number generator thing.
    + n307 = copy of 305
    + n308 = copy of 306
    + Problem: AE loss goes up once we start training the GAN adversarially…
        + 515: add enc net params to D opt
            + Works
            + Try again with lower LR for GAN
        + 516: add dec net params to D opt, n307
        + I'm training with different GAN nets......................
        + ~~__Best for AE loss seems to be to add ENC NET params to D optimizer__~~
        + Repeat with same net
        + 515: add enc net params to D opt, n307
    + 
    + 518: D learns enc weights, n307, 20k pretrain
        + AE loss reported by eval script is way higher than actual one… What does this mean??
    + 520: =518 with lower AE learning rate (2e-4)
+ Ok what if we train with no pretraining then?
    + 519: no pretraining
+ 521: n308 (no branch G)
    + AE loss still goes up according to eval script…
+ 522: n309 (no branch at all)
+ Why are we still not learning E-w correlations?
+ 523: n307 (original net), 10000e, 30k pretrain
    + __AE loss is reported high by eval script__
    + Figure out why
+ Try with smaller sequences. Can we make the GAN learn the correlations then?
    + 525: train_wgpt2, n310 (AE is one layer), sequence_length = 64
+ train_wgpt2, evalvae2, n311:
    + Don't separate p and w in G at any point in the training loop
+ n_critic. Enc adversarial step isn't done as often as recon step.
    + 530: n_critic=1

+ No pretraining: the autoencoder being trained against G should help induce correlations.
    + Can't remember the logic behind that but I did think about it…
        + __Despite my best intuitions, it doesn't work__
+ 536: n314 = simplest D/G possible: three convs (ks=1) and a lin
    + 2503686: 100,000e, train_wgpt2, enc_dim=8
+ 537: n315 = original net
    + 2503723: 100,000e, train_wgpt3, enc_dim=8
+ train_wgpt3 = hyperparameters from paper, adjusted to 2e-4 max
+ 2532613: normal data, 100,000e, n_critic=1
+ 2533506: n_critic=4, 10,000e
+ Both don't work. Okay we need to do something different…
+ Feed wire pos to the autoencoder
+ train_wgptxy + n316 + evalvaexy
+ Regularise code with real wire radius
+ 540: n316 (original net, one-layer AE), no pretrain
    + 2550375: 40,000e total
+ 541: n317 (original net, two-layer AE), some pretrain
+ 542: n317 (enc_dim=2)
+ 
+ Regularise AE on radius
    + 545: Still can't figure out E/w correlation..…
+ Back to no regularisation. Theoretically, GAN adversarial training of AE should make it so that the features and wire can be correlated. I just don't understand why that's not the case yet...
+ 2573851: 40,000e, n327, train_wgpt2, latent_dims=1024, enc_dim=8
+ 2573886: 40,000e, n328 (more gen capacity), train_wgpt2, latent_dims=1024, enc_dim=8
+ Made a difference in loss function: changing gradient penalty from taking raw generator output to enc(dec(G(z))) for wire
+ Makes quite a big difference in loss and rate of learning: applying hard gumbel before enc (587)
    + Soft gumbel (588): hard to tell if it's better or worse tbh
+ IT's not looking like GAN will ever learn E-w correlation…
+ Only linear in D, no sequence-wise convolutions, mean value at the output -> n331
+ Only G trains Enc, Only D trains Dec -> wire collapse
+ __Giving AE params to D tends to make AE loss go up__ -> D is screwing up G when it's able to affect both Enc and Dec
    + Let's try with only Enc params: 612
        + Looks ok
        + Retry with enc_dim=64 (was 8)
        + __D doesn't screw up G__, and wire distribution is learned fast (200e)
    + And with only Dec params: 613
        + Looks ok
        + Retry with enc_dim=8 (was 64)
            + __D screws up G__
    + And with both
        + D screws up G
        + Retry with enc_dim=64 (was 8)
            + __D screws up G__
    + __So D can only screw G up when it can alter the Decoder parameters...__
+ 610: n327, latent_dims=256, enc_dim=64. Seems promising. Only G affects AE. Looks ok
+ 615: n327, standard, enc_dim=8
+ 616: same as 615 but D can alter enc params
    + AE loss goes up during training -> Looks like disregard for AE
+ 617: same as 615,  enc_dim=64

+ Instead of embedding, we use a gumbel to index into the wire_to_xy array.
+ It's working...?
+ Wire interpolation is done by taking eps x seqlen of the real and fake sequence, respectively. Not the best way to do it but might be sufficient here?
+ 2649536: n340 (one branch conv in G and D), latent_dims=256, ~~enc_dim=8~~`
+ 2649539: same, normal dataset
+ 2649850: same, altered dataset, latent_dims=1024, ~~enc_dim=8~~
    + Hits are all concentrated around the center layers of CDC
+ 2649854: same, normal dataset, latent_dims=1024
    + Same, but stringing looks nice tho...

+ 651 vs 652: one overfits wires, the other doesn't
    + The difference is in architecture only: n346 vs n347
    + n346 increases feature maps before generating wire probabilities
    + n347 goes down to 256 feature maps and then Conv(256, n_wires)
    + n348: add an act+bn before passing wire probabilities to p branch
    + ~~2658344: n347~~
    + EVALVAE(NORMAL) 2
    + 2659641: n347, latent_dims=256	
        + Looks like it's learning wire distribution @2000…
        + Wire diversity is pretty meh
    + 2659718: n347, normal data
    + 2659627: n350, latent_dims=256
        + __Not__ learning wire distribution @2000
            + Doesn't learn wire distribution quickly @20000.
        + Concentrated on inner wires @6000...
    + 2659726: n350, normal data
        + Pretty bad wire collapse but samples are honestly not looking too bad	
            + Loss is stable, so this might be ok
+ Give G access to xy information: wgpt6, evalvae3
    + p branch determines features based on xy information
+ 2663787: n356, normal data
    + Inner wires. But loss is stable.
+ n359, n360: okay, we can sort of learn the W distribution by itself…
    + So why didn't it work before?
    + Try again with larger kernels
        + 674, n361 = same disc with k_s=3
            + Yup… Kernels make the net fail…
            + Probably because wire position isn't mean to be interpolated / averaged
            + But why does D think that it is smarter to discriminate this way???
                + What if we have 2-3 layers with k_s=1 before increasing?
                    + n363, 676: seems better, but still meh
                + n364, 677: +features, only k_s=1
                + n365, 678: decreasing feature maps in D
        + 675, n362 = same, with lighter D
        + n366: mean instead of linear, so we kill any sequential info
        + Decreasing features doesn't work. Increase and average -> n367
        + Summarise features and average over sequence -> n369 / 682
            + Okay.
        + n370: only wire
            + 683 -> Why is wire distribution so spikey???
            + 684 -> k_s=3 in final W branch layers G
            + made a mistake in n371...
            + 685 -> similar
            + 685 and 686 didn't work at all (n372, n373).
                + Both have same D and same as n371
                    + Both have gumbel tau=3	
                        + n374: tau=⅔ (687)
                    + Okay tau=3 might have messed us up.
                    + Stil, wire distribution is very NOT smooth. Can we make this better?
                        + Higher or Lower tau??
                            + 687: tau=⅔ 
                            + 688: tau=⅓
                            + 689: tau=1
                        + Increase or decrease feature maps before going to n_wires?
                            + 690: More layers decreasing feature maps in w branch, tau=1
                        + 691: k_s>1 in D
                            + seems bad
                        + 692: add r manually in D
                            + Doesn't make a difference @500. I'll train for more just to see
                                + It learns something, but not the right thing
                            + So it's the Generator's fault…
                        + 693: remove instance norms
                            + No change @500
                        + I had my xy_norm transformation wrong, so it makes sense that the inner wire were getting more attention, if. Let's see now.
                        + 695: n380 -> it learns...?
                        + n381: k_s > 1 in D
                        + n382 = n381 + DBlocks in D. 
                            + 697: seems to suck
                        + n383 = n381 + increasing feature maps in G w branch
                            + 698: no indication that it's any better
                                + Ez Wire collapse
                        + n384 = n383 but k_s=1 in D
                            + 699:
                        + n385: only one w layer in G, more feature maps overall
                            + 700
                                + 2681625
                        + n386: same with p
                            + 701
                                + 2681661
                        + n388 = 385 + DBlocks (w only)
                            + 702: 
                        + n389 = 385 + DBlocks + p
                            + 703
                                + Pretty good.
                        + n390 = Fewer weights in G, w only
                            + 704
                        + Increasing D layers doesn't seem to make W distribution drastically better @400
                        + n392 = n389 + no DBlock, only normal conv
                    + n394: simplified D, looks goodish -> 708
                    + n395 = w branch goes down to 1 feature map lol
                        + 709
                            + Okay, so it seems to take longer to train when we go down to 1 feature map before n_wires
                            + But is there an optimal number? Try 8
                    + n396 = convf rather than lin0
                        + 710
                    + n397 = one fewer instance norm, replace ReLU by Tanh in final w layer
                        + 711
                            + Wire overfit......? Or is it the lack of InstanceNorm?
                    + n398 = one more w layer, go down to 8 feature maps
                        + 712
                    + n399 = increase k_s in D top layer, add InstanceNorm back
                        + 713: hint of correlation @500? waiiiiiiiiiiit @1000… mmmmmmmmh @1500
                        + @2000 struggling still...
                    + n400 = p branch takes off from last w layer before n_wires, to increase correlation (?)
                        + 714: It doesn't
                    + n401 = tanh?
                        + 715: maybe it's just the tanh…
                    + n402 = one more layer in D, so that receptive field is 17x17x17=4913
                        + 716 __Dramatically increases the time we run for...__ Loss looks good tho
                        + Actually 17x17x7 would be enough.
                    + n403 = 17x17x5 in D
                        + 717: meh. Doesn't improve wire distribution or correlation at all, seemingly.
                    + n404 = cat instead of adding in G, replace D with 17x5x5x3
                        + 718: much faster. Net naturally seems to give more energy toward outer layers...?
                    + n405 = no tanh
                        + 719
                    + n406: __soft gumbel until n416__
                        + 720. Hmm, looks like it's hitting the outer layers all the time, whereas hard gumbel favours inner layers...?
                    + n407 = no k_s>1 in p branch
                    + n408 = no GBlock, just normal ConvTranspose+Norm
                        + 722
                        + Correlation is clearly apparent at the beginning. Soft gumbel is making most hits occur on outer layers, but can the net correct itself enough?
                            + Nope
                    + What about lower tau?
                    + n408 = lower tau
                        + 723
                        + 724 = n_critic=1, lower G lr.
                        + 2686879, n408, latent_dims=256
                    + Increase D lr ?
                    + n410 = don't go down in feature maps before n_wires
                        + 727
                        + Effect: wire dist looks like getting good @500e
                        + 2687673: 20,000e, n410, latent_dims=256
                    + n411 = one less layer in w and p branch G
                        + 728
                    + n412 = move BatchNorms to top of G
                        + 729
                    + n413 = go up in feature maps
                        + 730 -> tends to overhit same wire
                    + n414 = don't use xy in p branch G
                        + 731
                    + n415 = n412 + don't use xy in p branch G
                        + 732
                        + 2687934: 20,000e, n415
                    + n416 = n415 + hard gumbel
                        + 2688029
                    + n417 = n412 + hard gumbel
                        + 2688082
                    + n418 = D uses one-hot wire as well
                    + 737 = 735 + tensordot for real data as well (purposeful mistake (wire_to_xy))
                        + Clear blowup: D loss goes way down because real and fake data don't look the same at all.
                    + 738 = correct tensordor, make wire_to_xy not hit 1
                        + Still concentrates on inner wires at low epochs
                    + n420 = one fewer w layer to prevent overfit?
                        + 739
                        + 2704518 looks ok
                    + n421 = projection layers at top of D
                        + 740
                    + n422 = projection layer but no one-hot wire in D
                        + 741
                    + The more we train and the more G selects wires… What can we do against this?
                        + __HOW DO WE NOT OVERFIT WIRE?__
                        + n424: only xy and w features in D
                    + n423: lin instead of mean
                        + 742: loss looks strange at the start… INitialisation?
                        + __Does not learn at all.__ Combination of Conv( -> 1) and Lin is just horrible.
                    + n425: just lin0 at the bottom of D
                        + Weird loss for D, doesn't look like learning
                    + n426: only p features in D, lets' see what happens
                        + @100 not that great. Wire changes by itself, probably because p depends on it in G. Correlation between p and xy is apparent even if it is not correct.
                        + @500 Correlations slightly apparent.
                        + @900 Okay
                        + 2709376: Let's see what happens on the long term. Presumably it does well in the p features because it does even when we enable the w feature.
                            + @8000 p correlations almost perfect.
                    + n427: w and w_ohe, two linear layers D
                        + D is 270M params now
                        + @100 can't see much
                        + @500 w distribution starts to shift
                        + 2709481: Let's see at 10kepochs
                            + @3000 W distribution is close to learnt, but quite spikey. Does it get better or worse later on?
                    + n428: One more p conv G, only p branch D
                        + 747: p feature distributions quite spikey...
                    + What if G passes its w code to D?
                    + n429
                        + 748
                        + I'm retarded. We can't do this: how would real data have a wire code?
                    + 
                    + n430: n428 + Make p branch come out of upsampling convs instead of w code
                        + 749
                    + n431: n430 + DBlocks
                        + 750: compare with 749
                    + Oh I changed lambda_gp to 2. And then to 40
                    + Changed interp back to real interpolation	
                        + Doesn't look like a big difference…
                    + n432: Disable connection from xy to p in G
                        + 751: __Correlation between xy and p disappears in G output__
                        + __W distribution is uniform__
                        + But features look nice @500
                        + @1000e: distributions are starting to get real good correlations
                        + 2711280: 10kepochs -> __Let's see if it destabilises__
                            + @2000 very nice so far… Loss looks like D striving up toward 0 while G stumbles around going generally downward
                            + @5000 distributions and correlations are still very closely matched.
                                + G loss is going back up since 2000 
                                + D loss is extermely stable at ~ -6
                    + n433: n431 + activate xy in D. Same conv for p and xy
                        + 752: G loss shoots up @100
                    + n434: DIsable xy-p connection + activate xy in D. 
                        + 753: G loss shoots slightly lower @100
                            + @500: G loss goes up to 300. D loss stabilises.
                            + W distribution is off center, E-w correlation is nowhere to be seen, but same for other features.
                    + Randomise the xy / wg produced by G. Does the loss look the same?
                    + n435: randomised wire decision in G __but we still pass the xy info down to generate p__
                        + 754: G loss shoots up like when we don't randomise W…
                        + @500 W distribution is uniform. Very slight signs that outer layers get more energy?
                        + 2712018: 10kepochs
                            + @2000
                    + What if we reverse it? Use p info to draw wire choice?
                        + n436: Draw wire choice using p info, and check w branch gradients...
                    + 
                    + __When using the same seed, we often end up with the same wire at the output of G...__ Is that the case for p? Let's try
                    + 757: two layer with n_wires feature maps (but still projecting up)
                        + @500 wire distribution still spikey, but at least it's in the right spot.
                    + n438: increase feature maps so we don't have to project up in W choice
                        + We might need some x4 upsampling to limit parameters
                        + Let's go raw
                        + 758: 400M params in G
                    + n439: x4 upsampling in G 
                        + 759: latent_dims=64 We have even more parameters lol
                    + n440: only xy feature in D
                        + 2718656
                    + n442: use wire and xy
                        + 766
                    + n443: Interpolate instead of ConvT(4, 4, 0)
                        + Suddenly we don't overfit instantly?
                        + 767
                    + n444: add back p features
                        + 768
                    + n445 = n442 + soft gumbel
                        + 769
                    + n447 = convf instead of lin0
                        + 2719460
                    + n448 = only use wire (not xy)
                        + 2719414
                            + __ALMOST THERE..........?__
                            + Yes we already knew this kind of works, by itself, didn't we?
                                + -> cont'd 775
                    + n449 = only use p and xy
                        + 2719452
                            + Not looking so good
                    + Why so high gradient pen?
                        + It's when we use wire
                    + n450 = replace large convs with lins
                        + 774	: loss looks ok
                    + n451 = same with soft gumbe
                        + 776
                    + 779 = 778 + chunk_stride=8
                        + Clearly bad because of soft gumbel.
                + Increasing feature m		aps before wire choice does not magically solve our problem :'(.
                + 780 = back to normal
                + 781 = don't use soft gumbel. It doesn't make sense to do so and then tensordot with the wire_xy matrix…
                + 782 = 781 + chunk_stride=8
                    + Horrible wire collapse… Like bro it's worse than c_s=1???
            + So:
                + xy information tends to make G panic and its loss shoot up and the W distribution spikes all over the place
                + wireID information helps G to learn the overall shape of the W distribution.
                    + Not sure whether it helps with correlations -> check
                + p information alone is a piece of cake to learn both distributions and correlations
            + 783: only p and wireID info
                + W distribution is learned, but p distributions suffer quite dramatically so far @200
            + 784: gradient pen norm is only calculated over pxy (train_wgpt8)
                + Huge GP spike lol, but net seems ok and W distribution is learned FAST
                    + Blows up @200 :(… OR DOES IT?
                    + I changed it back between 100 and 200
                    + It eventually blows up.	
                        + But it learns W faster than with the naive GP
            + 785: weight clipping. n456 (train_wgpt7)
                + W distribution is learned FAST, but P suffers a LOT
                + Looks unstable as we increase epochs
            + 786: weight clip, n457
                + Ok wire but P is bad
            + 787: n458 = p takes two feature maps before convw
            + Net prioritises W distribution in n458-n460
                + Adding a p conv in G doesn't change that.
            + Maybe because D projects w to 256 dimensions while it doesn't do that for p
                + Can we balance them out?
                + n461: project p to 64, w to 256
                    + Yup. Now G starts caring about p features
            + Do we need to balance w projection in G and D? Give more to one or the other?
                + n462: only w branch in D -> experiment with projection size in both nets
                    + 790: still spikey @500
                    + 2724576: 5000e
                        + Fewer spikes @1000
                        + w dist bit spikey still @3500
                            + Wait I might have used wrong net (n461), so we're trying to learn p as well	
                        + __Honourable mention for trying to learn p and learning w successfully__
                            + p is far off though.
                + n463: more proj in G
                    + 792: compare with open tabs @200
                    + 2724589: 5000e
                        + Distribution almost perfect @4000. Some spikes and mismatch but ok
                + n464: more w layers in D
                    + 793: more spikey W distribution @100??
                    + 2724716: 5000e
                        + Much slower than the rest
                        + Tends to overfit for most frequent wire… @3000
                        + Pretty good @5000
                + n465: 463 + D wire projection has fewer dimensions
                    + 794
                    + 2724801: 5000e
                        + Hmm ok. Not the best.
            + How do we make the net learn the W distribution more faithfully?
                + Better D? Better G?
                + 
            + __Pass one-hot encoded wireID through wire_to_xy and ADD NOISE TO IT__
                + n466: add p back to n463
                    + 795: more fm in w than p -> p not great @100, w ok
                    + 797: more fm in p than w -> still p bad and w ok
                        + 2725275: 5000e
                            + p explodes. W is fine but one wire is too active. @5000
                    + n468: one more common conv G
                        + 799: No weights @100. Missing Norm layer. Vanishing grads.
                        + 799: retry with BatchNorm.	
                            + Better but p very bad
                            + Weird developments… @200
                            + Is G too weak?
                        + 800: n469 = only p
                            + No bueno
                        + 801: Add a batchnorm before p projection, but don't activate
                        + 802: taper down feature maps in G
                        + 803: fewer fms in D for p
                            + still not…
                            + Maybe it just learns more slowly…
                            + Actually it looks quite a bit better than the previous ones… n471
                        + 804: 
                        + 805: activation before p proj in G
                        + 806: activation in p proj in D
                        + 807: more DBlocks
                        + 808: More strided conv in G
                            + Check @500
                                + Not great actually… It explodes...
                        + 809: remix of n471 to try to make it work. Let's see @500. 
                            + Blows up
                        + 810: one more p conv G
                            + not looking great tbh
                        + 811: k_s=17 in D
                            + Okay first hint at being not TERRIBLE
                                + Huh so the discriminator can't actually tell much if we activate without passing it through a large kernel type layer…
                            + Wait it lost its cool @500
                                + Tendency to explode
                        + 812: original net (466) with k_s=17
                        + 813: Convs in G
                        + 814: back to one conv in p branch in G
                        + 815: 
                        + 816: sigmoid activation before projection in G
                        + 817: gp `train_wgpt9.py`
                        + 818: ONLY P. gp separated between p, xy and w
                            + __p works with GP, but not with weight clip__
                        + 819: nvm
                        + 820: p and w in D n484
                        + 821: only w
                            + __wire gradient penalty shoots at 100 instantly. Why?__ Can we just ignore it in our GP term? I think it's preventing everone from learning.
                        + 822: 818 repeat to see gradient pens
                        + 823: only  w. Ignore w GP term
                            + Catastrophic loss curve.
                        + 824: divide w gp by 3606, w only
                            + Ok @100
                                + __Learns wire distribution__
                        + 825: same, p and w. 
                            + Hmmmm. Loss jumps around. W GP term still goes up to 1e2
                        + The problem is when we combine the w and p features and gradient penalties… Why is the w GP term so large ??
                        + What if we give it the real wire sequence everytime?
                            + 826: still goes wild
                        + 826: the urn n484
                            + 2727959: 20,000e	, latent_dims=64
                                + Destabilises around 10k epochs and does not learn xy -> E correlation
                            + 2744938: 20,000e, same but latent_dims=256
                        + 829: one more p conv in G, n487
                            + Can see p features appear @500 already
                            + 2728189: 20,000e, n487
                                + Turns out to be unstable @14000…
                                + Can never learn 
                                    + Figure out if it's caused by latent dims or net arch:
                                    + 2735460: same, n487, latent_dims=64 (like successful runs)
                                        + Does not learn E correlation
                        + 828: without /3606: w gp term is high at first but goes down. Stays around 2 for a while
                        + 
                + n467: add xy back to 466 and add a bit of noise to it
                    + 796: w > xy > p
                    + 
                    + 798: w = xy = p: w ok, p very bad
                    + 831	: with xy, latent_dim=64, n489
                        + 2728581: 20,000e, latent_dim=64, train_wgpt9
                            + Careful of destabilisation @17,000?
                                + What happens after that?
                            + 2743297: 20,000 more epochs
                                + ??? It reverts back to uniform edep. I don't understand...
                    + 832: with xy + noise(0.01), n490
                        + 2728608: train_wgpt9
                        + 2774449: same with latent_dims=256
                    + __WE DID IT...?__ Check by reversing the edep dependency
                        + Can we fix up the last distribution: edep per layer, before the meeting?
                    + n491 = one more p conv in G
                    + 2734736: 20,000e
                        + Looks like it's going in the right direction @3500
                            + Becomes unstable much faster than 489/490
                            + D is identical to 490
                    + 2734949: same, normal dataset
                    + n492 = more feature maps in D
                    + 2734817: 20,000e
                        + Bit unstable @1500. GP is going up! But it goes back down later according to n491.
                            + Instability is probably caused by extra p convs
                    + 2734943: same, normal dataset
                        + Not very good features (E, dca distributions too thin)
                    + n493: add some noise (0.02) to wire one hot encoding
                        + 834
                    + n494: x2 upsampling layers in G
                        + 2744897: 20,000e
                            + Looking ok @2500
                            + Goes unstable @3000
                    + 836: more noise in W and xy proj in D, + train_wgpt10 (larger grad pen for w)
                        + 2745031: 20,000e
                            + GOes unstable @6000. xy-E correlation isn't learned
                    + 2774275: n496 = n490 (the urn) + no extra p convs
                    + 
                    + n498 = p and w are concatenated in D before projection. wgpt11 = instead of interpolating features, we do the same as the rest and concatenate a subsequence from real and fake
                        + 2782570: 10,000e
                        + 
                    + train_wgpt12: don't transform xy, only rescale
                    + 2784242: 10,000e, n498
                        + 
                    + 843: latent_dims=8
                    + 2790419: wgpt13, n499, ld=100
                        + Wire collapses
                    + __CHANGED wgpt9: initialise D with fan_in and do not scale down w GP__
                    + 2794890: wgpt9 but we don't scale down w GP, and we initialise with fan_in
                        + 2794890: n502, 10,000e
                        + 2795033: n490, 15,000e 
                            + @3500 p and w distributions slowly being learnt
                                + Correlations not very apparent yet
                    + train_wgpt9_altered2: reverse energy correlation with wire
                        + 852
                        + 2796043: 15000e, n490, LD=100
                            + … Not seeing it.
                        + 2804967: 30000e
                            + __OH WAIT WE KIND OF ARE SEEING IT @30000__
                        + 2854155: another one (45,000e)
                            + Still stable, still good
                        + 2907459: another one (60,000e)
                            + Still going but samples aren't really getting any better…
                        + 2948503: another one (90,000e)
                            + Instability coming around @65,000
                    + We have gradient penalty for w and xy, but technically they are the same thing.
                        + If we calculate xy internally we won't have xy as an input to D so we can't calculate gradient. Does that improve GAN? Make it worse?
                        + 
                    + train_wgpt14 + 503 = don't pass xy to discriminator, let it calculate itself using the matrix
                    + Manual gradient scaling for wire?
                        + When we pass backward through the wire matrix, the gradients close to the center are divided by small numbers so they increase?
                    + 861: more p convs with k_s=3
                        + Oh. It works… What if we add the convs before splitting the branches?
                            + 862, n504: distribution is more pointy.
                        + What if we make them k_s=1?
                            + 863, n505: distribution more pointy as well
                        + Three p __and__ w convs:
                            + 864, n506: same. Looking like it's distributing energy non uniformly across wires? @500
                                + Nope, nvm @1000
                            + 2856269: let's see what it can do at 15,000e, LD=100
                                + Gorgeous distributions @8500. Wire uniform, as expected.
                                    + Still great @15,000. Does it ever get unstable?
                                + 2875843: 60,000e. Let's see.
                            + 
                        + For real data, we could tensordot wire_xy and detach to make it a leaf variable instead.
                        + n507: pass xy to disc but not w
                            + 865: Inner layers get more hits @500e. Grad norm?
                                + Also the case @1000 but it's going up
                                    + @2000 wire starting to look collapsing
                                + Add 3,000 epochs
                                + @5000 wire collapsing, G loss is going up since @2000e
                                + 2867541: 20,000e total
                                    + 
                        + train_wgpt16: wire gradient rescaling
                            + 866: no rescaling, n507
                            + 867: rescaling ON!, n507
                            + 2867567: 20,000e
                                + G loss goes up @2000 and wire collapses.
                                + Still collapsed @20,000
                                + TODO: check whether wire collapses before that point
                                    + Not completely collapsed yet but were getting there quite fast.
                                    + Why do we collapse wire so quickly?
                                + Can we figure out what gradients are causing the increase?
                            + n508: branch convs in D, replace InstanceNorms with BatchNorms
                                + 869: grad divided by wire radius
                                + 870: grad multiplied by wire radius
                                    + __Wire looks more central after 500e__
                                        + Loss looks identical…
                                    + @1000: wire distribution spikey
                                    + @1500 Spikey Spikey
                                + 871: grad multiplied by 1000
                                + 872: hooks
                                + REDUCE LR by factor 10
                                + 2924363: wire grad divided by radius (wgpt16), n508
                                    + gradients of xy, w, p printed out
                                    + Wire Collapsed
                                + 2924722: wire grad multiplied by radius (wgpt17), n508
                                    + Wire not as collapsed but too low
                                + So there's a factor 10^3 difference between the gradients in G's bottom convw and convp…
                                + But this difference isn't present in D's top convw and convxy...
                                + n509: only w in D
                                + n511: instancenorms in D
                                    + 2965950: 15000e, train_wgpt18 -> wire gradient increased
                                + train_wgpt17: debug printouts and hooks everywhere
                                + train_wgpt18: remove all printouts, multiply wire grad by sum(wire_r_norm)
                                + 873
                                + n512: one more conv in branches D
                                + train_wgpt19: replace xy by r, theta. It makes more sense for gradient propagation if the dataset is heavily based on r.
                                    + 2966057: 20,000e, n508
                                + train_wgpt20: don't rescale wire gradients
                                    + 2966084: 20,000e, n508-> compare with 2966057
                                + 883: multiply grad by wire_to_rth norm
                                + wgpt21: CDC sphere
                                + 884, n511
                                    + 2999580: 10ke -> no bueno
                                + 886, n508
                                    + 2999579: 20ke
                                + Forgot to put in wire_sphere… So 2999579 and 2999580 are still using normalised rtheta transformation
                                + __WIRE SPHERE__
                                + n513: =n508 with wire sphere
                                + 3006865: n513, 20,000e
                                    + @20,000 better than @10,000
                                    + Wire collapse not too bad but not as good as when we have wg as a feature.
                                + 3034292: cont'd 40,000e
                                    + spikey. I'd say distribution is better than @20,000 but not by much...
                                + n514: only w
                                + 888:
                                    + @1000 incraesed G lr by factor 8
                                + 3035853: increased LR
                                    + __GOOD__ wire distribution @20,000e. Not spikey.
                                + n515: no branches in D
                                + 889
                                + wgpt22_normal: we randomly reverse the order of interpolated sequences.
                                + 890
                                + 892: n514 (only w), random reverse in interpolation
                                + 893: n514 (only w), normal data
                                + 894: n513
                                + 3047182: 20,000e
                                    + Wire dist is quite spikey
                                + 895: n513, normal data
                                + 3047183: 20,000e, normal data
                                    + __Promising__, but spikey wire
                                + n516 = one more conv in w branch in D
                                + 896: 
                                + I want to make those spikes disappear because they simply mean that a specific wire is getting hit in too many samples.
                                + n517: reduce G conv layers
                                    + 897 -> nope @2000
                                + n518: reduce feature maps in w branch more
                                    + 898 slightly better but @1000
                                + n519: k_s=65 in convw1
                                    + 899 less spikey
                                + n520: k_s=5 in convw2
                                    + 900 quite spikey and slow to learn
                                + n521: reduce feature maps even more in w branch
                                    + 901
                                + n522: k_s=3 in convf in D
                                    + 902 seems better but more slowly
                                    + 3054223: 20ke
                                        + spikey wire @5000 & @10000
                                + n523: with noise in first xy and p layers.
                                    + 903
                                    + 3054966: 20ke
                                        + spikey @6000
                                        + So noise doesn't fix the problem… What in the hell does?
                                + __chunk_stride experiment: GAN doesn't seem to learn anything more if we train with e.g. chunk_stride=8 and 1/8th the number of epochs...__
                                + bababababbabababababa
                                + __n524 + train_wgptpac = PacGAN__
                                    + 905
                                    + 3059164: 40,000e
                                        + Good but wire is way too spikey
                                    + 3059192: 40,000e, normal dataset
                                        + Doesn't seem to make a difference with respect to wire collapse...
                                    + Let's not use it but I'll keep the code anyway
                                    + Back to train_wgpt22 + n523
                                + wgpt23 + 525: one-hot w feature in D
                                + 3093256: 40,000e
                                + 3094270: 40,000e, normal dataset
                                + n526: more convs in w branch in both nets. Make D architecture better reflect G architecture
                                    + 3119730: 20,000e
                                        + 3155378: 40,000e
                                    + 3119741: 20,000e, normal dataset
                                        + 3170300: 40,000e
                                            + __VERY NICE__
                                    + Very tail of wire distribution isn't very very good. Let's try one more w layer in D
                                + n527: one more w conv in D
                                    + 3155714: 20,000e
                                        + 3170378: 40,000e
                                            + Very very good @40,000, except very tail of wire/E distribution
                                    + 3155737: 20,000e, normal data
                                        + 3170377: 40,000e
                                + A geometry where every point has the same norm and the distance between two points in theta is the same as in phi.
                        + wgpt24: CDC cylinder instead of sphere surface.
                            + We need the right spacing between layers in the z plane
                            + And we need radius to not stray too far from 1
                            + 3172480: 20,000e, n527
                                + 
                            + 3172492: 20,000e, n527, normal dataset
                                + Very nice @20,000
                            + 3174909: 20,000e, n526
                                + Very nice
                            + 3174920: 20,000e, n526, normal dataset
                        + wgpt24_testcylinder = debugging and nice 3d plots
                        + 909, 910
                        + wgpt25: Fix cylinder phi offsets
                            + 3179071: 50,000e, n526
                            + 3179073: 50,000e, n526, normal
                            + 3179075: 50,000e, n525
                            + 3200570: 50,000e, n527
                            + 3179076: 50,000e, n527, normal
                                + 3200549: 100,000e
                                + 3271010: 150,000e
                                    + Still going strong. Some trouble reaching the innermost/outermost layers.
                            + 3241848: 50,000e, n527, normal
                                + Good
                            + MADE A MISTAKE IN 527: WE'RE NOT CONVING AS MUCH AS WE THOUGHT IN G
                            + n528: fix and add convolutions in D and G
                                + 3202506: 50k, n528
                                    + 3240223: 100k
                                        + Meh @100k. Wire bad.
                                + 3202510: 50k, n528, normal.
                            + n529: fix n527 but don't add anything
                                + 3203077: 50k
                                    + 3240228: 100k
                                        + Big meh @100k. Wire distribution not learned well.
                                + 3203067: 50k, normal
                            + n530: more minimal D and G
                                + 3213602: 50k
                                    + ~~3240229: 100k~~
                                + 3213612: 50k, normal
                            + wgpt26: manually put the cylinder on a radius=1 sphere (see output_912)
                                + 3242863: 50k, n527
                                    + Has bad wire distribution and correlations
                                + 3242958: 50k, n527, normal
                                    + Good distributions. Wire struggles on innermost and outermost layers.
                                + 3242887: 50k, n530
                                    + -> smoother/better p distributions than 527 @4000
                                    + Struggles with wire distribution and correlation
                                + 3242977: 50k, n530, normal
                            + wgpt27: Konst's third dimension
                                + 3248314: 50k, n530, R=R_max+800
                                    + Not very good -> can't go farther than intermediate layers
                                + 3249558: 50k, n530, R=R_max+800, normal
                                    + Prioritises inner layers, but otherwise ok
                                + 3248337: 50k, n530, R=R_max+200
                                    + Worse than +800
                                    + Has trouble going beyond intermediate layers
                                + 3249545: 50k, n530, R=R_max+200, normal
                                    + Prioritises inner layers like +800
                            + wgpt28 + n523 = no w feature	
                                + 913 -> still wire collapse, no bueno. We need wire feature.
                            + n531 = n530 + dropout
                                + 3266457: 20,000e, wgpt27
                            + n532 = n531 + geom conv + residual in D + small G k_s
                                + 3267710: 20,000e, wgpt27 (konst)
                                + 3268122: 20,000e, wgpt26 (cylinder sphere)
                                    + 3289961: 40,000e
                                        + __VERY VERY NOICE__
                                        + 3366024: 80,000e
                            + wgpt29: konst + divide x,y,z by variance
                            + Might not be the norm but the variance that needs to be 1…
                            + wgpt29 = manually set wire xy variance same as p variance.
                                + 3275505: 20,000e, n532
                                    + Works @20,000e. Wire E correlation not the best. Slightly favours inner layers.
                                    + 3335903: 50,000e
                                        + Getting better at correlations @30,000. We do have hits in all wire IDs, which is nice.
                                        + Wire distribution worsens with epochs now @45000
                                        + Not good @50,000e
                                    + wgpt34: Same data std as not-working experiments, to check. Still three dimensions (z=0)
                                    + n540: n527 + more convs in D
                                        + 3384796: 20,000e, wgpt34, n540
                                            + Looking worse than 5505 @10,000e
                                                + So close @20,000e… Seems to be working despite different normalisation. Perhaps the problem is going to two dimensions???
                                            + 3404741: 40,000e
                                                + Very nice @40,000e
                                        + 3385052: 20,000e, wgpt34, n540, normal
                                        + 3387510: ^, n532 like 5505
                                            + big oscillations in G loss @10,000
                                            + OK @20,000 but inner layers favored slightly.
                                    + n541 = n527 + more dropouts in D
                                        + 3391562: 20,000e
                                            + __Nope__. Dropout doesn't do it.
                                            + Almost recovers well at the very end of @20,000
                                + 3280721: 20,000e, n527
                                    + Works…
                                    + 3316865: 40,000e, wgpt29
                                        + Same loss as for full sphere -> good
                                        + high wire ID suffers @35000…
                                        + Looking better @40,000e
                                        + 3345015: 60,000e
                                            + Still seems to be learning @60,000e
                                        + 3366282: 100,000e
                                            + Nope. Looks like a bug really. It only hits inner layers
                                + wgpt35: rescale gradient based on norm (+wgpt29)
                                    + 935
                                    + 3393193: 20,000e, n532, wgpt35, __NO HOOK__
                                        + Nice @20,000e
                                    + WE'RE NOT APPLYING THE HOOK
                                    + 3404626: apply the hook, n532
                                        + __DOESN'T WORK__ @8000
                                        + Works @20,000e. Inner layers stil prioritised despite hook.
                                        + Loss looks okay though
                                    + 3404627: same, normal
                                    + 3404628: n527
                                        + Still prioritises inner @9000
                                    + 3413151: continuation of 3321774 @40,000e (wgpt29) with wgpt35
                                        + Note: this is a hybrid run. It might explode.
                                    + 
                                + 
                            + n536 = n532 + dropout 15% -> 5% + some other conv changes
                                + 925
                                + 3300352: 40,000e, wgpt29 (variance set)
                                    + Okay. Somewhat struggling with outer layers initially
                                    + but doing ok @25,000
                                    + G loss goes into different mode @30,000
                                    + Trouble putting all big hits in outer and all little hits in inner layers…
                                    + Struggling @40,000e.…
                                    + 3344070: 60,000e
                                        + Tendency to hit inner layers more than outer.
                                + 3321774: 40,000e, wgpt29_normal, n536
                                    + Let's do it
                                    + Looking ok @7000
                                    + Nice @40,000
                                    + Best so far. CM33 results. Still selects inner layers more than outer… __investigate__
                                    + 3406333: 60,000e
                            + n537 = n527 + dropout 5%
                                + 3317033: 20,000e, wgpt23 (full sphere)
                                    + Very very good except in very tail of wire/E distribution. (As in muec update 22/02)
                                + 3347230: 20,000e, wgpt29 (variance), n537
                            + wgpt31: replace z by radius
                                + 3342508: 20,000e, n537
                                    + Large radius struggles, possibly because farther away...?
                                        + Slowly crawls out @8000
                                        + Okay @20,000e. Still not very good with outer layers and correlations.
                                    + 3365627: 60,000e
                                        + Okay-ish @60,000e
                            + wgpt32: no third dimension. Proper std normalisation to continuous feature's variance like wgpt31, G lr=2e-4.
                                + n538 = n537 + no third dimension
                                    + 3347441: 20,000e, n538, wgpt32
                                        + What is it learning? @16000
                                + n539 = n527 + no third dimension
                                    + 3367193: 60,000e
                                        + Not learning the right stuff… Same as 3347441
                            + What's the std for the sphere gemoetry?
                                + wgpt23: [0.5027, 0.5027, 0.6982]
                                + So it's actually larger than data std [0.1929]
                                + And largest on the z-axis, which represents layer/radius
                                + wgpt32 (x-y): [0.1929, 0.1929]
                                + What if we make the radius' std-dev the same as the features' std dev?
                                    + wgpt33: radius std is same as feature std
                                    + 3368020: n539, wgpt33, 50,000e
                                        + Not good at all
                            + wgpt34 with n540 works well, but it still uses a 3-dimensional wire_sphere… Can we make it work with 2-d representation?
                            + __wgpt36 + n542__: 2-d wire-sphere
                                + 3453361: 20,000. __Gradient pen is wrong__
                                    + Works
                                + Wire sphere STD in both dimensions is now equal to p STD
                                + 3456698: 20,000e. __Correct gradient pen__
                                    + Strikes inner layers @8000
                                    + Works @20,000, very similar to no grad pen...
                            + What if our training set is one sample?
                            + wgpt36_one
                            + What if our generator sucks?
                            + n543
                            + What if our disc sucks?
                            + n544
                            + 
                            + Difference between training on one sample and 64 samples: wire distribution is better represented.
                            + Wire distribution converges very very quickly compared to P distributions… Can we help it?
                            + __JOB 969__ THIS IS WHAT COLLAPSE LOOKS LIKE. Very cool
                                + __n554, gan36, altered_one__
                                + So both losses suddenly spike and our sample converges on the training sample.
                                + However, spatial relationships and wire/feature correlations aren't good
                                + Can we maybe improve disc so that the collapse is better?
                            + n555: sum instead of mean in D
                            + 971: n555 with real data
                            + DOesn't seem to work as well…
                            + 972: 554 with real data
                            + 973: n556 = n64=8 in D
                                + Still does it
                            + 974: n557 = more convs and larger k_s in D
                            + So our net is learning to hit the right wires but not necessarily in the right order
                            + We overfit faster when we increase G width
                            + 978: n512=1024 in G
                            + 979: n512=2048 in G
                            + 980: two samples
                            + n560: fewer branch layers in G
                            + 981
                            + n568: pxy
                                + 990
                            + n569: upsample nearest
                            + 1001: nice
                                + __Maybe the key is smaller embedding, or no activation…__
                                + Works initially @5000 but then falls down @40,000
                            + n579: strides in D
                                + 1004
                            + TODO: Try smaller embedding in D and no activation in WGAN-GP case
                            + 1008: no xy
                            + 1011: no w
                            + 1015: p and xy
                            + 1016: p and xy, more emb_dims
                            + 1017: p and w
                            + 1018: emb_dim=1


                    + Gradient penalty in xy.
                        + Makes sense or no?
                    + For real data
                        + We need to focus heavily on the xy patterns.
                        + Maybe have a branch with parallel convs of various kernel sizes which we then concat 
                        + 
                        
                    + Start experiments from n598 (1038):
                    + In what order to we concat / add wg, xy and p to get best training?
                        + n599: concat wg and xy, then wemb
                            + 1039 -> Learns wg very very fast
                        + n600: xyemb, then cat and tanh to x
                            + 1040
                        + n601; add xyemb and wgemb
                            + 1041
                        + n602: only xy
                        + n603: +xy emb
                            + 1057
                    + XY feature isn't doing anything… What is the GAN learning?
                    + 1058: 1 epoch
                    + Okay. We finally got an improvement in wire distribution with only xy (1078)
                            + gan37.py = wire sphere has R
                        + First solution uses wire x, y and r, as well as wire ID in wire sphere
                            + -> gan38.py, n614, 1078
                        + Can we do it with only r?
                            + gan37, 1077
                                + Meh @2000
                                + Not showing @4000
                        + Can we do it with only wire ID?
                            + gan39, 1079
                                + No signs @2000
                                + Struggle @4000
                                + Collapse @6000
                        + Can we do it with only x,y?
                            + 
                        + Does ReLU speed it up?
                            + n616, 1080
                                + Looking a bit spikey @2000
                    + Try adding layer ID?
                    + Try shifting to 0 mean
                        + 1082: performance looks same
                        + 1083: divide by gu.n_wires rather than n_wires.
                            + Same
                        + 1084: n617 = increase ngf to 32 before n_wires
                            + Looks better
                        + 1085: n618 = reduce emb dim in D 256->32
                            + Works. Much less spikey
                        + 1086: n619 = n617+ increase ngf to 64 before n_wires
                        + 1087: n620 = n618+n619+ more feature maps in G
                            + Spikey
                        + go back to mean+conv+lin instead of lin+lin
                            + n621, 1088
                                + __Much much better__
                                + G params: 1,55M
                                + D params: 0.84M
                                    + Starts to get spikey @4000
                            + Before, D has 3M
                        + 1089: n620 + Fewer D feature maps
                            + bit worse
                        + train_gan40 + n623: add layer ID to geom tensor
                            + 1090: Similar to 1088. Not sure if better.
                        + 1091: More W convs in G
                        + 1092: larger k_s in G w convs
                            + n625, no layer ID
                            + Okay, but spikes don't magically disappear
                        + 1093: More upsampling convs in G
                            + G now has 0.7M params, D has 0.8M
                            + Performance good.
                        + 1094: 10 latent dims
                        + 1095: 1 latent dims -> G has 0.65M params
                        + 1096: reduce feature maps in G -> 31k params
                        + 1097: reduce feature maps in D: -> 235k params
                        + 1098: n629=same with tanh in G
                        + 1099: n630=tanh before gumbel
                        + 1103: n632=very small tau
                        + 1104: 
                        + 1105: very large tau
                        + 1108: increasing tau
                        + 1109: decaying tau
                            + __BETTER__
                        + 1110: dims 2 and 3 of geom tensor are zeros
                        + 1111: remove norm shifting in geom tensor
                        + __Result of experiments__
                            + Trouble learning wire distribution when we don't provide additinoal dimensions to geom tensor
                            + TODO: Retrieve jobs that show this and save figures. 1078, 1077, 1079
                        + 1116 vs 1115: more G params makes learning faster but we see instability as well.
                            + Still wire collapse @2000
                        + 1117
                        + 1118: 
                        + 1119: bias=True
                            + Seems better
                        + 1120: k_s=5 in G
                            + Looks same
                        + 1121: more D feature maps, k_s=5
                            + Very different loss. D goes down to zero faster and spikes a bit.
                        + 1122: reduce D feature maps
                        + 1123: reduce D k_s
                        + 1124: maxpool in D
                        + 1125: no spectral norm (n641)
                            + Doesn't work
                        + 1126: zeros in geom dims 2 and 3
                            + __Doesn't work__
                        + 1127: only wire ID in geom tensor
                            + Works
                        + 1128: only R in geom tensor
                            + Works…
                        + 1129: geom_dim=3
                            + Works (n642)
                            + So what did we change that made it work? G ndf? Max pool? Branch upsamples in G?
                        + 1130: reduce G ndf
                            + Less good but works. (n643)
                        + 1131: normal dataset
                            + Meh
                        + 1132: no noise to geom tensor
                            + Still works
                        + 1133: no maxpool
                            + Loss looks fine.
                        + 1134: no padding in D
                            + Same
                        + 1137: xy embedding without spectral norm
                            + Works. (n647)
                        + 1138: reactivate maxpools. __COMPARE__
                            + Pretty bad peaks in losses at the start
                        + 1139: add p back
                            + Not very good
                        + 1140: only p
                            + Loss is pretty flat everywhere
                        + 1141: k_s=1 in D everywhere
                        + 1142: k_s=64 taper in D 
                            + Distributions look smoother but loss is still meh
                        + 1143: strides in D
                            + __Pretty GOOD, VERY SMOOTH__ and loss looks nice. (n652)
                            + So by increasing receptive field of D, we smoothen up the distributions. Does this occur with xy as well?
                        + 1144: residual connection with concat
                            + Loss isn't as pretty
                                + Ah I made a mistake, I interpolated the wrong tensor
                        + 1145: fix up residual connection
                            + Loss looks terrible
                        + 1146: boop up D feature maps
                            + __same problem__ Feature maps in D don't help
                        + 1147: linear interp
                            + __BETTER__	. Downsampling with linear interp is better
                        + 1148: area interp (adaptive avg pool)
                            + Seems better for learning and loss curve
                                + Correlations appear @4000
                        + 1149: actual adaptive avg pool to compare
                            + Okay, lets' use that (n658)
                        + 1150: back to xy only
                            + Meh
                        + 1151: area interp, xy
                            + Nop
                        + 1152: No residual, xy
                            + Loss peaks but structure appears. Maybe D is too good?
                        + 1153: Fix activation
                            + Same problem in loss (n662)
                        + 1154: xy noise
                            + Still bad peaks
                        + 1155: no noise but dropout
                            + Still bad peaks (more?)
                        + 1156: fewer D maps
                            + Still...
                        + 1157: more dropout
                            + Still
                        + 1158: higher dropout p (n667)
                        + 1159: more feature maps G
                            + More peaks, worse results
                        + 1160: fewer feature maps G (512) n669
                            + Not convincing
                        + 1161: even fewer (256) n670
                            + Not convincing
                        + 1162: more D feature maps
                            + Meh
                        + What's the difference between n647 and n672 that makes w work or not?
                            + G are the same -> Can only be in D
                            + Embedding?
                            + Downsampling?
                            + Kernel size?
                        + 1163: xy embedding
                            + Nope
                        + 1164: no downsampling
                            + It learns but quite badly. Still spikes
                        + 1165: k_s=3 in D
                            + __And now it's not as spikey anymore.__
                            + Why 3?
                        + 1166: k_s=12
                            + Nope
                        + 1167: k_s=1
                            + Worse than 3
                        + 1168: k_s=5
                            + Works but loss spikey (n677)
                        + 1169: last w conv is 9, 1, 4
                        + 1170: one more w conv 17, 1 ,8
                        + 1171: 33, 65
                            + Still hella spikey
                        + 1172: same with 64 in D
                            + Spike city
                        + How do we save the loss curves?
                        + 1173: n677 + beta1=0.2
                            + Worse
                        + 1174: beta1=0.9
                            + Seems less spikey but maybe slower learning?
                            + Continue training...
                        + 1175: more k_s in D, beta1=0.9
                        + 1176: lr=5e-5 for D
                            + Seems to help… Not sure if it learns a lot of stuff
                        + 1177: only p
                            + Big meh
                        + 1178: n652 (worked for p), with new training params
                            + Still meh
                        + 1179: same but beta1=0.5
                        + 1180: more stride
                            + Loss doesn't move
                        + 1181: one more conv
                            + __Much much nicer loss for p__ n685
                                + Only difference is k_s and one more conv in D. Surprising...
                        + 1182: same as 1179 but lr is back to 2e-4
                            + Nicer loss as well…
                        + 1183: rerun of 1181 with lr=5e-5 to make sure. beta1=0.5.
                            + Good loss. Push training.
                        + 1184: xy, n686
                            + myeh. Bad looking spike
                        + 1185: D beta1=0.9
                            + Slightly better
                        + 1186: both beta1=0.9
                            + No better
                        + 1187: beta1=0.5, D lr=2e-5
                        + 1188: G lr=5e-5
                        + 1189: G lr=8e-4
                            + Ultra spike.
                        + 1190: G lr=2e-5
                            + Still spikes
                        + 1191: beta1=0.9
                            + D loss goes to zero instantly.
                        + 1192: lr=4e-4 D, 2e-4 G
                            + Looks bad
                        + 1193: Reverse
                            + Learns but really badly… D loss is zero
                        + 1194: more strided conv in G
                            + Bad collapse
                        + 1195: 
                            + Bad collapse
                        + 1196: balance feature maps n689
                            + Bad collapse
                        + 1197: p
                            + __Works much better with beta1=0.5 and lr=2e-4 in G/D__
                            + One or the other?
                            + Only lr=2e-4
                                + Explodes
                            + Only beta1=0.5
                                + Goes nuts on extremes
                        + 1198: xy, k_s=3 in D
                        + 1199: same as 1196 with 1197 params
                            + Not good
                        + 1200: n691 (k_s=3) and try changing hypers
                            + Then go back to n689
                        + 1201: beta1=0.01
                            + Still spikey
                        + 1202: Rmsprop D
                        + 1203 beta1=0.5, lr=6e-5
                            + Seems better
                        + 1204: beta1=0.7, lr=2e-5
                            + Nicer curves
                        + 1205: p
                            + Work-ish
                        + 1206: beta1=0.9
                        + 1207: p
                        + 1208: both
                            + Kinda works
                        + 1209: residual xy
                        + 1210: more residual xy
                        + 1211: more residual p
                        + 1212: more k_s D
                        + 1214: fewer maps G
                        + 1216: crazy conv top of D
                        + 1217: 3, 1, 1 conv in G
                            + nope
                        + 1218:less maps in D
                        + 1220: bigger k_s in D
                            + Learns p
                        + 1221: xy (n704)
                            + meh
                        + 1222: no reducing maps in D
                        + 1223: 1221 + G(512)
                        + 1224: G(1024)
                        + 1225: reshape
                            + no difference
                        + 1226: D(more k_s)
                        + 1227: D(512)
                        + 1228: D loss twitched
                        + So loss is pretty similar between k_s=3 and k_s=8, but k_s=8 loss goes to zero much faster…
                        + 1234: geom_dim=6
                            + 
                        + 1235: D(32), k_s=8
                        + 1238: ok
                        + 1239: simplified D
                        + 1240: p
                            + 
                        + 1241 = 1238 + p only
                        + 1242 more G maps
                        + 1243 +1 G layer
                        + 1244 +1 G layer and BN
                        + 1245: xy
                            + Works
                        + 1246: both
                        + 1247: geom_dim=6 (x, y, r, theta, wireId, layerId)
                            + wireId isn't geometrical
                            + layerId is redundant with r, I think
                        + 1248: 
                        + Generator residuals: use interpolate, try nearest and linear
                            + 1254: One residual in G
                                + Improvement?
                            + 1257: xy, x4 upscales, geomdim=3
                            + 1258: linear interp
                        + Still don't know why xy training loss gets spikey as we increase D k_s.
                        + 1259: wgan xy
                            + no better
                        + 1260: p
                            + Seems to work. Better correlations than standard GAN
                        + 1261: increaase feature maps in D out of bottleneck
                        + 1263: no strides D
                            + Different loss curve but no imrpovment
                        + 1264: geom_dim=1 (only r)
                            + Nope
                        + 1265: convw with k_s=1 convs
                            + Nope
                        + 1267	: gradients_w instead of gradients_xy penalty
                        + zero grads in last ~5 elements in sequence consistently
                            + 72120 zero grads in total, and 72120= 5 x 3606 x 4(batch)
                                + Padding problem. Padding=8 fixes the issues
                        + 1282: no residual D
                            + No work
                        + 1283: no residual G
                        + 1285: lr=1e-4 -> learns p
                        + 1286: residuals
                            + Learns faster
                        + 1287: w
                        + We can learn via either w or xyr… But we overfit very quickly…
                            + Needs @3000 to see w distribution change…
                        + 1290: tau=1 from 0.75
                            + Overfit less pronounced but still showing.
                        + 1291: try with xy grad
                            + Much faster learning of xy distribution, but GP is quite high…
                            + Is GP as high for p?
                            + Do we overfit?
                                + Yes @7000
                        + 1292: noise
                            + Seems same
                        + 1291, n730 = xy
                        + 1293, n731 = p
                            + p grad pen and xy grad pen are on different scales. Why?
                                + Maybe because the raw D loss is much lower so the net doesn't care about grad pen as much.
                                + So maybe the question is: why is D loss so low for xy?
                                + So critic rates real samples on the same level for p and xy but fake samples for xy have very very low score… Same at large epochs.
                                    + That's probably why G struggles to learn.
                                + D somehow knows how to differentiate real from fake samples straight away. Why? How?
                            + Do we also overfit p when we get to ~7000?
                                + Nope. Distributions keep getting better.
                        
        + Plot: for each wire, the average energy of hits received by that wire.
        + __Need to figure out what layer to have before the gumbel choice so that our wire distribution is a smooth as possible__
    + Write down how many epochs on average we need to get satisfactory results…
        + Only features ->
        + Only wire -> 
        + Both -> 500e for distributions
            + Way more for correlations

+ wire pos -> [encode] -> encoded vector -> [decode] -> 

+ [ ] Determine the best candidate among `train_` scripts. 
    + [ ] AdamHe (no AE gradient descent) -> cannot consistently learn wire distribution
    + [ ] AdaBelief -> Unstable
        + [ ] AdamHeAEoptAE (AE gradient descent, D/G affect AE)
        + [ ] AdamHeAEoptLR (AE gradient descent, D/G train independently)

+ **Problem: the AE trains very slowly compared to the rate at which the GAN learns.**
+ [x] **Either pre-train it or figure out a way to make it better**
+ [x] **Try more layers**
+ [x] **Try different optimizer/lr**
+ [x] **Try pre-training but care of overfitting**
	




   + In-your-face demonstration that correlation are learnt by GAN
       + Make all high-E hits go in x plane or something
		

Huh? np.unique(data.wire) shows that not all wires even get hit once? can we ignore this or is it very bad?


   + +++ **wganae44**: make AE learn on all wires at a time
   + -> ae_states_v3.pt
       + L1
            cross entropy loss: 0.008004397829063236
            accuracy: 1.0
            dist loss: 0.0047394853178411725
            norm loss: 0.004869550787843764
            norm mean: tensor(1.0054, device='cuda:0', grad_fn=<MeanBackward0>)
            norm std: tensor(0.0696, device='cuda:0', grad_fn=<StdBackward0>)
            norm min / max: 0.853844940662384 1.164057731628418
        + L2, 50e -> ae_states_v4.pt
            cross entropy loss: 0.000235321282671066
            accuracy: 1.0
            dist loss: 0.0006900659232633188
            norm loss: 0.0002828737773234025
            norm mean: tensor(1.0028, device='cuda:0', grad_fn=<MeanBackward0>)
            norm std: tensor(0.0166, device='cuda:0', grad_fn=<StdBackward0>)
            norm min / max: 0.9557589888572693 1.0659523010253906
            tensor([   0,    1,    2,  ..., 3603, 3604, 3605], device='cuda:0')


- [x] Check if AE loss changes much after we let the GAN train…
- [x] Add distance reg to AE pre-training

- [x] Try: training AE alone with CrossEntropyLoss. Eventually, it should give better results for our particular problem.
- [x] [ ] We did it. But it needs pre-training.
   
- [x] MedGAN uses TanH in Enc and Sigmoid in Dec. Compare with ReLU.
    - [x] Looks better like medGAN
- [x] [ ] Not always true. In altered dataset experiments, Tanh/Sigmoid is bad

+ [x] Add some plt.close after each plt.savefig


- Still have to figure out:
    - [x] Best optimizer and learning rate for AE
    - [x] How many layers required? 2 okay?
    - [x] How many encoding dims? 16 okay?
    - [x] Do we still use 513 kernel_size? Try TCN?
    - [x] Does n_critic have any impact? Yes

- [x] Note: wire position interpolation for gradient penalty… Since the CDC geometry has an empty center, what are we really doing???
- [x] So the difference between using 100% real_w and 50% real_w/fake_w doesn't seem to matter after 200 epochs… Still haven't found the right formula for this...
- [x] We need to figure out how to increase wire diversity. All samples hit the same 8 wires all the time.

+ noise_level experiment (15 vs 1390944) -> Result *looks slightly* better on **zero noise**
+ tau experiment (1400003 vs 1390944) -> Faster tau decay distributions are slighty worse
+ weight decay experiment (1400004 vs 1400005) -> Weight decay bad with spectral_norm but okay if we remove it

+ nn.ModuleList: properly registers Modules when they need to be stored in a list.

   + **TODO**: 
+ [x]  plotify
+ [x]  Read Brenninkmeijer master's thesis
+ [x]  Build icedust on lx00
     + [x]  Failed: missing ncurses-libs
+ [x]  Get it right.
     + [x] [ ]  Meh
+ [x]  Plot time difference in eval plots.
     + [x] [ ]  **ALMOST THERE**


+ To implement:
    + [x] Could try strided convs in disc again, but they seemed to suck last time…
    + [x] Lines between consecutive hits in scatter plots, so order can be seen

[][][][][][][][][][][][][][][][][][][][]
  [ ] [] [ ] [] [ ]
  
+ [x] Use TCN design to make a causal generator/disc:


   + 1291, n730 = xy
   + 1293, n731 = p
       + p grad pen and xy grad pen are on different scales. Why?
           + Maybe because the raw D loss is much lower so the net doesn't care about grad pen as much.
           + So maybe the question is: why is D loss so low for xy?
           + So critic rates real samples on the same level for p and xy but fake samples for xy have very very low score… Same at large epochs.
               + That's probably why G struggles to learn.
           + __D somehow knows how to differentiate real from fake samples straight away. Why? How?__
               + Can we make it suck a bit more?
       + Do we also overfit p when we get to ~7000?
           + Nope. Distributions keep getting better.
   + 1294: smaller kernels
       + G score is still super low
   + 1295: Remove a bunch of layers in D
       + Okay, G score goes up but slower to learn @1000
       + Still overfits @6000
   + 1296: add arange(n_wires) to geom features
       + Overfits even faster.
           + !! But then recovers!
   + 1297: latent_dims=500
       + Looks same
   + 1298: Linear layers in D (n735)
   + 1299: geom_dim=1 arange(wires)
       + Overfits very heavily…
   + 1300: convolutions instead of Lin
   + 1301: flatten instead of mean
       + Gradient pen has huge peak at the start...?
   + 1302: circular padding (n739)
       + Same as 1300. Doesn't fix overfit problem…
   + 1303: D n512=1024
       + Large spike
   + 1304: D n512=256
       + Nice @2000, but is it a fluke?
       + First spikes @6000
           + It's probably a fluke.
       + Overfit @10000
   + TODO: figure out how spikes can be avoided.
   + 1305: D n512=128
       + Spike @6000
   + 1306: G linear w
   + 1308: same as 1306 with n512=512 in D
       + Spike @4000
   + 1307: bottleneck (4) in G
   + 1309: bottleneck (1) in G
       + Spikes @6000
   + 1310: fewer maps G
       + Spikey @8000
   + 1311: n512 to n_wires (n748)
       + Smoother @2000 (too smooth)
       + Ends up spiking but later @8000-10000
   + 1312: more convs in D
   + 1313: More maps in G
   + So is it smooth because we make G dumb or because it's working...?
   + 1314: +1 layer G
   + And we're back at the start.…
       + Very bad score for G, which gets better when it starts to spike heavily. WHy?
   + 1315: dropout in G and D
   + 1316: lower temp gumbel
   + 1317: soft gumbel
       + Not much different
   + 1318: p to see
   + Keep training 1316
   + 1319: softmax
       + Nope
   + 1320: np.arange(end, start, -1) in geom tensor
       + @1000 Figures out distribution with wrong wire IDs (Slopes are reversed on discrete ranges)
       + @2000 Spikes like a mofo
   + 1321: no mean subtraction
       + Same
   + 1322: no mean subtraction, normal arange
   + 1323: lr=5e-5
   + 1324: lr=1e-3
   + 1325: flat geometry
   + 1326: no standardization
       + Doesn't learn anything in both cases.
   + Keep training 1325:
       + No improvement. One spike???
   + 1327: Layer ID in geom tensor
       + Step-function like
           + -> Step-function like in wire distribution
   + 1328: gan45w, n757, layer ID + theta (geom_dim=2)
       + Smoother
   + To try: 
       + very large latent dims
       + more maps from geom_dim in D
   + 1329: arange(wire) with no normalisation
       + Critic output and gradient pen magnitudes go way up, but the results are similar.
       + Having norms from 0 to 3606 does not cause imbalance in distribution, it seems
       + Peaks @3000
   + 1330: very large latent_dims=4000
       + Spikes
   + 1331: geom_dim -> 1024 conv in D (n758)
       + No spikes @2000
       + Maybe higher #features in D out of the input?
   + 1332: geom_dim -> 2048
   + 1335: 50 noise
   + 1336: 500 noise
   + Why doesn't noise make any difference?
   + Re-add standardisation
   + 1338: 0.1
   + 1339: 1.1
   + 1340: stop at minimum in G loss (@750)
   + Try: Linears in D
   + 1341: noisy geom
   + 1342: Linears
       + Ok @4000
       + Spikes
   + 1343: D acts as if there's no sequence
       + Nope. Very spikey
   + 1344: Only block linear layers
       + No explosion @8000 but wire dist is flat…
   + 1345: One more lin layer
       + Has trouble learning w dist.
   + 1346: GBlocks in w branch
       + Similar results…	
       + ??? peak in first wires???
   + 1347: One more lin D
       + ??? peak
   + 1348 (n766) No GBlock but more lin in D
       + No better…
   + 1349: conv 9 top of D
   + 1350: fewer maps in first conv
   + 1351: geom_dim -> geom_dim in first conv	
       + Still spikes
   + 1352: conv k_s=3
       + No spikes @30,000 but no learning either
   + 1353: same, more fmaps
       + Now spike @2500
       + So maybe. The positioning of hits is too random in the sequence, so convolutions are actually hurting instead of helping.
       + But then why can't linear layers learn the true distribution of wire ID?
       + What if we boost capacity of first conv layer by massive amount?
   + 1354: only lin, p and xy
       + Hmmmm
   + 1354: only lin, xy, more feature maps
       + Smooth 
   + 1357: No seq-wise lin in G.
       + dips at wireID maximum...
   + 1358: Fmap increase in G w branch
       + Same but worse dip
   + 1359: more lin in G
       + Same dip, no improvement really
       + @10000 Finally reverses the dip but spikes also pop up
   + So changing G doesn't seem to do anything…
   + 1360: Big conv top of D
   + 1361: sort of encoder top of D
       + Ok dip in the right direction…
       + Spikes…
   + 1362: no bulk lin in D, sum bottom of D
       + Quite good @2000
           + Collapses on most probably wire it seems
   + 1363: GBlocks in G
   + 1364: same as 1362 with no mean subtraction in standardisation
       + No difference. __Performance is independent of Standardisation__
   + … So Sum helps?
   + 1365: more conv
       + Noop
       + Wait yes @3500 but spikes
   + 1366: more fmaps in first conv
   + 1367: InstanceNorm in D
   + 1368: LayerNorm on last two dimensions
   + 1369: LayerNorm only on feature dimension
       + Spikes
   + 1370: No LayerNorm, sum before final lin
       + No difference. Makes sense.
   + 1371: More fmaps in D	
   + 1372: Fewer fmaps in D
   + 1373: InstanceNorm affine=True
   + 1374: n512=512
   + 1376: geom_dim -> geom_dim
   + 1377: no 9, 1,	 4 conv
   + 1378: mean instead of sum
   + 
   + 1379: More Conv/InstanceN layers (n792)
       + Weird behaviour… Artifacts in loss
   + 1380: sum instead of mean (n793)
       + Strange loss as well
   + 1381: flatten then lin
       + Loss peaks at first
   + 1382: same but no weight init
       + Much more reasonable loss initially.
   + 1383: sum, no weight init (n793 again)
       + D Loss spikes up like a bastard
   + 1384: mean, no weight init
       + Better, some learning
   + 1385: k_s=3 in D
       + Spikes @8000
   + 1386: k_s=3 in G last layer
       + Spikes
   + 1387: k_s=1 in D, decaying fmaps
   + 1388: n_critic=1
   + Well, I had xy+noise(0, 1) all this time...
   + Okay. It doesn't work. Maybe the bottleneck is just too small…
       + Let's go back to passing around one-hot wire
   + train_gan46w.py
   + 1389
   + 1390: k_s=3 in D
   + 1391: disc requires grad setting
   + 1392: no k_s=3 in last G w branch
   + 1393: same as 1392 but disc always requires grad
       + Spikes @3000…
   + 1394: 
   + 1395: no InstanceNorm
       + Much smoother loss and better distribution, straight up
   + 1396: increasing fmaps
       + Still bad loss curve
   + 1397: Remove last InstanceNorm
       + Still not great. Peaks in GP
   + 1398: Increasing fmaps AND no InstanceNorm
       + Alright. Good dist @2000. Can we improve?
       + Still good @10000
       + Still good @20000
   + Residual connections in D -> n805
   + 1399: 
       + Works.
   + Add p back (n806)
   + 1400: Residuals and p
       + Fat explosion
       + Ah I forgot to do gradient pen on p...
   + 1401: No residuals and p
   + 1402: only w
   + 1403: p and w + __Gradient pen correction__
   + 1404: p and w __+ ACTUAL gradient pen correction__, n809
   + 1405: normal dataset, latent_dims=100
   + Now that GP works, try: residuals, geom tensor
   + 1406: D residuals (n810)
       + ya
       + Okay @10,000. Some correlations apparent. Maybe improve G?
   + 1407: More GBlocks (n811)
       + \Same. Maybe it;s the dataset that's too small (should have thought about that first...)
   + 1408: larger dataset (full altered).
       + Definitely better distributions
   + 1410: add xy (n812)
       + Okay. Good news is that it doesn't explode
       + Bad news is that it's not learning wire / E relationship (@40,000)
   + 1411: project geom tensor before concatenation (n813)
       + GP spikes…
           + But doesn't explode. Keeps learning @12000
       + @20,000 signs of learning E/wire relationship
           + But distributions lose in quality as well....
   + Try: initial projection to smaller space to increase geom_dim importance
       + project xy and concat to wg tensor, or vice versa..?
   + 1412: n64=8
       + Looking similar to 1411.
   + 1413: fmaps never go down (n816)
       + Spikes less pronounced than 1411
   + 1414: project p and wg to 8 fmaps and concat xy 
       + Spikes much less pronounced. Loss looks a bit like 1410.
       + However E/w correlation isn't well-defined at all…
   + 1415: only project wg to 8, concatenate with raw p and xy
       + Okay @17000. Correlations not so apparent yet.
   + 1416: more info in geom tensor: radius, theta (gan48w, n819)
       + Quite spiky loss. 
   + I think projecting xy is necessary, otherwise xy relationships cannot be properly convoluted over with the same kernels as P. The projection layer should maybe be like an encoder.
   + 1417: xy projection to 32 fmaps (n820)
       + Still quite spiky…
           + Do we need multiple convs in the projection?
   + 1418: one more conv in geom projection (n821)
       + Better
   + 1419: one more (n822) conv, fmaps is n64 as well.
   + I had an activation layer after projection...........
   + 1420: one projection layer, no activation. (n823)
   + 1421: same with n64=32
       + Looks worse...?
   + 1422: n64=4
       + Less spiked…
   + GP gradient with respect to wire_sphere?
       + gan49w
   + Fix standardisation to bring geom tensor into better range
   + 1423: n825
   + n826: one projection layer for p, xy and w
       + 4023848
   + n827: one for p, one for w, three for xy
       + 4023880
       + Perhaps better here?
   + 1424: Three projection layers for w, p and g (n828)
       + @10,000 correlations become apparent
   + 1425: InstanceNorm in G (n829)
       + Okay…
       + 4042931
           + 
   + Big difference: G from CM33 didn't have strided convs in branches. Only one (3, 1, 1) conv
   + On the other hand, D had k_s>1 convs in branches…
   + 4043001, 1426: Common branch strided convs in G. Avoid uneven strides.
       + Does not fix the problem @15,000
       + And neither @40000
   + 1427: One proj layer. (n831)
       + 
   + 1428: one LayerNorm in D
       + Similar results @1000
   + 1429: more LayerNorms
       + Oo nice it works @1000. Maybe use this? __n832__
       + Oh wait it's blowing up @2500 __NOPE__ abort mission
   + 1430: One proj layer, better k_s in D (n833)
       + Worse...
   + 1431: nearest in D residuals instead of area
       + Uh ok huge difference.
       + __very very bad__
   + 1432: No interpolation in first layer
       + Much better but
       + GP spiking like hell
   + Maybe mix w and g first, then p
   + 1433: mix w and g first
       + not much better
   + 1434: dropout(0.2) on p-w mixed output
       + Better! Much less spikey @2000. Perhaps later
       + Still going @5000. Very nice.
           + Finally spikes @10,000
       + So maybe D spikes when it's too certain about samples.…
       + Indeed 536 had dropout as well	
           + However it had no residual in G
       + What made 536 work and ours not?
   + 1435: dropout(0.5)
   + 1436: no w/g mixing before p cat
       + Big GP Peak
   + 1437: dropout in second layer instead of first
       + Better	
   + 1439: divide by max norm instead of standardizing.
       + Spikes and Explosion
   + 1440: two more layers and instancenorm in xy projection
   + 1441: nearest in G (n842)
   + 1442: k_s>1 convs in top of D
       + It doesn't do it.
   + 1443: no xy (n844)
       + Unstable still...
   + 1444: revert
       + 
   + 1445: no residual D, dropout(0.1) (n846)
   + 1446: add xy back, 4 convs with increasing fmaps to project
   + 1447: more convs for w branch	 D
   + 1448: High k_s common branch convs in G (n849)
   + 1449: noise (0.01) to wire sphere (n850)
   + Forgot bn in G
   + 1450: with bn in G (n851)
       + 4065893
           + It knows…
               + __It knows @37000__
                   + However quality of w and p distributions takes a hit.
           + What makes it know?
               + Conv(1) proj layers before mixing w and p?
                   + 4068286: only one proj layer for p, g and w (n855)
                       + Very nice results @40,000. Distributions are clean-ish and it somewhat figures out geometry.
                       + __4085982__ Keep going
                           + Still stable @80,000
                       + __4106586__ another one
               + Absence of too many strided convs in D?
                   + 4068059: two more stride=4 convs in D (n854)
                       + __Yup that seems to be a problem__
                           + __So is it the strides or the fmaps?__
               + Presence of two more common convs in G?
                   + 4068498: remove common convs in G (n856)
                       + Still works quite nicely.
               + wire_sphere noise?
   + 1451: one proj layer (n852)
   + 1452: make wire_sphere range from 0 to 1 on all axes (train_gan51w, n852)
       + Then adjust wire_sphere noise level
       + __IT KNOWS AS WELL__ (@14000 apparent)
       + 4068892
   + 1453: noise(0.05)
   + Let's try to add back some stuff to 1452 and see if it still knows
       + 1454: Two more stride=4 convs in D
           + It doesn't know...?
           + Aaaaand it's blowing up @15000
       + 1455: one k_s=9 conv and more stride=2 convs in D
           + It seems to struggle… Not apparent @15000
       + 1456:
           + Struggling too. Not apparent @15,000. Explosion incoming
       + Maybe the difference is that we have multiple Conv(1) projections
       + TRY
       + n858 + more conv1 projections
       + Also, in n851, the conv(1) projections for g are increasing in fmaps. (maybe just a detail)
       + n860: three convs per branch
       + 1457
           + Not apparent @15000
           + Barely apparent @20,000, or maybe I'm biased…
       + 1458: 	Only one p proj layer.
           + Seems to be happening @15000, despite more stride conv layers. Nice!
               + Actually not that much better @20,000. Either learning really slow or not good enough… What are we missing from before?
           + What if we have more residual? Can we make it learn faster?
               + n862
       + 1459: residual connections in D
           + It knows something's up @7000
       + 1460: More common branch convs in G (n863)
           + No big difference @15000. Actually 1459 seems to fare slightly better in the edep vs layer distribution.
               + If only I could just nudge it over…
           + Struggles to move over. @30,000
           + It should be done by now, but let's go to 40,000
           + 
       + 1461: normal
   + 851 and 852 work, but it stops working after later changes. It's plain to see on the loss and GP curves actually. There's a clear difference. What causes it?
       + Too many strided convs?
       + Jobs wont run.…
   + 1462: like 1452 but with only one proj layer. Let's see if it knows. (n855)
   + 1463: Same as above but with gan51w (geom tensor in (0, 1) range)
       + Works. __It's not the geom tensor__
   + 1464: n856
   + What if we keep conving and doubling fmaps but then immediately conv(1) back to 512 or something? Does loss change shape?
   + 1465: equivalent of n855 but geom_dim is 1 (wire_ID) (n864)
       + Still seems to work!
   + 1466: k_s=3 in D convolutions (n865)
       + Still works! Or does it @15000?
           + Yes it gets better. @25000 we have decent result like 1465
           + @30,000 it looks like CM33
   + 1467: 2 more stride=2 convs, bring fmaps back to n512 with conv(1) afterwards
       + Yup we see the change in loss curve and GP. Not sure why this happens… Try dropout?
       + __IN ANY CASE, THE PROBLEM COMES FROM STRIDED CONVS IN D__
           + However we still see the edep vs layer distribution shifting…
           + It still looks like it wants to get better @25000
   + 1468: Dropout in all conv layers (n867)
       + Loss is more straight.
   + 1469: n866 with real dataset
       + We see the same loss / GP rising as in the altered set.
   + Wait I need to be more careful with 1466. At 15000, it looks way worse than 1465.
   + No it's fine.
   + With 864, are we clear of the GP pattern?
   + 1470: n864, real dataset
       + Observe GP and loss compared to 1469
   + 1471: real dataset, full data, n864 (minimal D, only 3 strided convs)
       + ___4091611___
           + Weird artifacts in distributions
   + 1472: small data, smaller G, n868
       + Still good @75000
       + We're still going strong @250,000 epochs… Nice.
       + 4128207
           + G loss slowly shoots up @300,000
   + ___4091491___: real dataset, full data, n866 (more substantial D with 5 strided convs)
       + __Note:__ No geom info in geom tensor (only wire ID)
       + ___4135691___ Keep going
   + Changed how dataset_altered_one restricts dataset size:
       + Previously was done during chunking
       + Now is done during data alteration
   + Check gradients of loss with respect to w.
   + Test with only w, only g, the magnitude of gradients in w tensor
       + n869 = n864 with hook
       + 1474
       + p+w+g: 7.2e-5, 1.0e-4
       + p+g:      4.3e-5, 6.9e-5
       + p+w:     6.5e-5, 9.6e-5
       + 1475: p+g
       + 1476: p+w+g (n870)
       + 1477: p+g, plot p grad, 15k e
       + __1478__: p+w, 15k e
           + __It knows without geom tensor…__
       + 1479: only k_s=3 convs (no stride) n872
   + 1480: xy, r, theta in geom tensor. n873 (three strided convs). train_gan55w
   + 4136774: ^ + full dataset
       + 4162493: continued. __Very good so far @120,000__
       + 4210665: 200,000e more
           + Nice distributions
           + 4444890: another one
           + 4552530: contd: Is it going to get there eventually? Or is it overfitting to single samples?
           + 4633923: contd
               + Even @1,000,000 distances aren't the best
   + 1481: much more complex D. 7 strided Conv layers. (n874)
       + 4161371. Able to learn dataset quite well (in overfitting fashion) @200,000
       + 4178618: contd.
           + Pretty much perfect. Four samples learned to perfection @350,000
       + 4190140: full real dataset
           + 4464069: contd -> don't seem as good @300,000.
   + 1482: Dblocks with residual connections and 1x1 convs (n875)
       + 4162148: small (real) dataset
           + 4211295 cont'd (on ful dataset)
               + Diversity suffers.
       + 4162154: full (real) dataset
           + Very slow… Results seem better than 140 (@60,000) @100,000
           + 4250632: cont'd
           + 4444811: contd
           + 4552518: cont'd
               + Not really learning distances @500,000
+ Dropout just before softmax?
+ Train on real CDC dataset.
+ Try again with xy passed externally and GP applied to it
+ train_gan57w.py, networks894.py
    + 4626524: n897 (interpolations and residuals), full dataset (train_gan57w_full)
        + Not great. D loss goes flat and G loss goes up
    + 4626623: same, altered (full) dataset	
        + Correlations not so pronounced. GP goes up.
+ External xy but no w?
    + train_gan58w -> no w GP
    + n898
    + 1510: 
        + Learns E-w relationship but w distribution is pretty bad and not getting much better
        + Oh wait we were using xy as a leaf tensor in GP… is that bad? What about in D loss?
    + 1512: same with noise in wire_sphere
        + Only slight difference in distributions (p as well). w is still super spiky
+ External xy, no w in D, w in GP but no xy in GP:
    + 1513: n898
        + Doesn't work (flat w distribution)
+ p, w and xy in GP but xy isn't a leaf tensor. -> gan61w
    + 1514 
        + … Doesn't work?
+ only w in GP, w and xy in D
+ D / GP: w | xy | both
+ w      |       |      | 
+ xy     |  o   |  1515  | 
+ both |       |             | 
+ 
+ ~~1515: xy in D, xy in GP
    + ~~yes @10,000
+ 1515: xy in D, xy in GP, not leaf in GP
    + Works @10,000
+ 1516: make xy in GP leaf
    + yes @10,000, but results slightly different from 1515. Hard to say if we're having any effect.
+ 1517: xy is GP leaf with .detach().requires_grad_()
    + Yes. Hints @5000
+ 1518: use passed in interp_xy
    + Yes
+ 1519: add w back (to GP), recalculate interp_xy after interp_w.requires_grad_()
    + Doesn't work @10,000
+ 1520: only xy in GP
+ 1521: all (p, w, xy) in GP, n897
    + Should get old results
+ 1522: detach interp_xy and make it leaf in GP (gan62w)
    + GP doesn't seem to go up too much @5000. Lets' go for more
    + So GP goes up until ~30,000 and then back down. Can we do something with this?
    + 4686519: contd to 200,000
        + Looks very nice @80,000 however wire distribution isn't perfect.
        + Very nice
+ Maybe the problem is that when we compute the gradient wrt w, we propagate through the xy branch as well, so the gradients through xy are (accumulated) twice?
+ Either way one of the gradients is off…
+ 1523: only w in GP, supposedly accumulates for both xy and w.  (gan61w)
    + __Explodes (both w and p)__ @5000. Clearly xy GP isn't enough.
+ Lots of epochs, but check @5,000 as well
+ I messed up 1523. Was using xy in GP. __ONLY W IN GP__
+ With the grad function, are we populating the gradient of interp_xy? Check with a hook or something.
+ 1524: do not detach xy and use w gradient only
    + Good but trouble learning E/w relationship (compared to 1522)
    + w gradient is sufficient to get w distribution right, but not geom relationships
+ __For altered data to work, we need a gradient penalty with respect to xy__
+ gan63w: detach xy and use w gradient only. Same as 1524 with detached xy
    + Explosion
+ So if we use w gradient only, we need to keep xy attached
+ Make a decision:
    + Detach xy in GP and take gradient with respect to it. (like gan62w, 1522)
    + But there's the fact that gradient wrt w also propagates through the xy branch…
    + No it doesn't. We pass xy and w directly to the
        + It does, but only in the G step
        + Does it make a difference if we detach xy in the D step?
    + 1526: detach xy from w in the D step, both for real and fake (gan64w)
        + Do we get the same loss curve as 1522/4686519?
        + Not exactly the same loss but results are pretty much identical. So detaching in D makes no difference.
        + __It's only a GP issue__…
        + 4695004: continued 200,000
            + Very very nice. p distributions are looking like they're training on their own. GP is looking pretty constant. Losses have converged around the same value and oscillate about that. Continue?
            + 4713252 contd to 400,000e
            + 
    + Detach makes a difference in the gradient of D with respect to real samples…
        + Gradient wrt __w__ is larger when we don't detach… Makes sense I think
        + … but gradient of convg0 and convw0 stays the same regardless of detach
    + 1527: weight clipping (c=0.05)
+ Is GP allowing us to learn E/w corr?
    + We can't really check because we can't turn GP off without the training blowing up.
+ Do autograd.grad params make a difference?
+ retain_graph: True mandatory otherwise we can't call grad() multiple times
+ only_inputs: 
+ 1529: only_inputs=False
    + Looking similar to 1526
    + Both 1526 and 1529 look worse than 1522 at the same epoch. Why?
        + Maybe it's just about the turning point so the difference is most noticeable.
    + 4699708: contd to 100,000
+ 
+ What changed between the converging loss curves of 1522 and the diverging / flat D ones of 4626623?
    + gan57w vs gan62w
    + The detaching of xy in GP? Seems so from diff
    + 
+ Compare performance between detach and no detach in D output
    + No detach: 29.6s (100e)
    + Yes detach: 29.7s (100e)
+ Compare performance between only_inputs True and False
    + False: 29.6s (100e)
    + True: 29.5s (100e)
+ No performance impact… So is there an option that gives better results / loss?
+ 
+ Real data: can we also get the loss turnaround as we do in 1522 and 1526?
+ 1531: real data (small) (gan67w)
    + Looks like it gets distance
+ 4705184: contd to 100,000
    + Looks very good @60,000
    + Very nice @100,000 as well.
+ 4707109: same with no xy detach in GP (gan68w). Expect to not work (losses don't converge back)
    + __Wait it works. What?__
    + Ok it seems not to be the detach thing then… 
    + It seems the only difference between loss convergence and non-convergence is 	
        + 1. Applying grad pen to xy
        + 2. Dataset size (small vs large)
    + With small dataset, losses turn around by epoch 30,000 and then meet back around @100,00
        + This is not the case for full data
    + So is the net's capacity too low then?
    + Wait for job 4707172 and see if losses converge
        + Doesn't seem like they are @50,000
+ 4707172: full dataset (gan67w_full)
    + 
+ What about no detach on altered dataset?
+ 1532 / 4717356: gan68w_altered (no detach, altered data)
    + Still works quite nicely
+ 
+ Can we make the turn-around faster by increasing either G or D capacity?
    + gan69w = gan67w + small altered dataset
    + 1533 / 4731125: n899 (n512=1024 in G)
        + Increasing G capacity seems to make the edep vs layer distribution a bit better.
            + It makes the loss reunion occur much faster…
            + Very nice @100,000
    + 1534 / 4731184: n900 (n512=1024 in D)
        + Much slower, but in the end, good results. Last layer still not quite there.
    + 1535 / 4731698: n901 (n512=1024 in both)
        + Quite good. We have edep in last layer, but next to last isn't so good.
            + Very similar to 1125 (only 1024 in G)
+ 4731864: train_gan67w_full + n899 (double G capacity)
    + Meh @60,000
    + 4838974 contd	
        + Getting better very slowly @160,000
+ 1536 / 4738348: split stride=4 convs in G into stride=2 to increase n_params to 22,838,169 (n902) (+ n512=1024 in G)	
    + Good except in last layer...
+ 4739613: ^same, full real dataset
    + Meh @100,000
+ 1537 / 4751193: n903=n902+ n512=2048 in G. n_params=77,162,777
    + Close to perfect @200,000
+ 4751855: same, full real dataset
    + 
+ 4782686: same, full altered dataset
    + Pretty much perfect by @12,000
    + Goes wild afterwards. What's happening?
    + It's not bad just a bit odd...
+ 1538: small real dataset, n903
+ Maybe more filters?
+ n904: more filters in each branch G
    + 1539
+ 1540: linear (n905)
+ 1541: (n906): less p, more w in G
+ 1542 / 4843471: (n906): fewer G params for speed, correct activations for GBlocks
    + Big meh @50,000. G loss goes way up.
    + Does not recover @100,000
+ Maybe better filters in D. What do we need for this specific dataset?
+ 1543: small dataset: 4 samples, stride of 2
+ n907: one more filter in DBlock, more params
    + 1544: 4861527 (small dataset)
+ dataset_small2: 16 samples, stride=2
    + 1545 / 4861529: n906
        + 4873342 contd
            + Getting quite good @200,000
        + 4929365 contd
    + 4861530: n907 -> Struggle is real @30,000. Samples are noisy mess
+ n908: not as many downsamples in D, one more filter in g and w branches.
    + 1456: small data
    + 1457: tiny data
+ n909: merge w and g into geom, apply different k_s and then cat
+ n910: small kernels, no DBlocks
    + p distribution suffers?
+ What about a mixed architecture where we do downstrides for p and something different for w?
+ Or, use dilation in w branch
+ n911: dilations in geom branch
+ 1550: gan67w
+ 1551: gan70w (batch_size=2), otherwise minibatch represents the whole dataset…
+ 1552: dataset = 8 samples, stride=1 (small3), batch_size=4
+ 1553: n912
+ 1554: n913
+ 1557: n915 (residual DBlocks, one in each branch, then one common one. No dilations, but one stride=4 conv.
    + 4879037: small3
        + Quite good
        + 4890667: contd
    + 4879065: full
        + 5018893 contd
+ 1558: n916 (more filters in common branch)
    + 4879345: small3
        + Gets messy @100,000 (D loss spikes)
    + 4879349: full
        + Struggles @30,000
        + Not great @42,000
+ 1559: n917 (k_s=3 filter after each downsampling conv)
    + Good
+ 1560: n918 (k_s=3 filter right after cat)
    + Good
+ 1561: n919 (918 + n64=4)
+ 1562: n920 (915 + n64=4)
+ 1563: n919, tiny data
+ 1564: n920, tiny data
+ 1565: n921 (n919 + D n64=8, G n64=8)
+ 1566: n922 (n906 + n64=8 in D and G)
+ 1567: n923 (n921 + n64=128)
    + better in W
+ 1568: n924 (n922 + n64=128)
+ Maybe the answer is in increasing capacity for bigger dataset…
+ Let's see if we can progressively increase the dataset and how the net reacts.
+ n925 (n915 + one more k_s=3 conv at end of D) D_params=6M
+ 1569: tiny data
    + 4905622
+ 1570: small3 data
    + 4905704
+ 1571: small4 data (train_gan71w) -> 16 samples
    + 4907276
+ 1572: small4 data, n926 (double ndf) -> D_params=27M. Honestly hard to see the difference @1000.
    + 4907287 -> clearly better than smaller equivalent @8000.
        + Looks better @14000 too
+ 1573: n927 (double ndf again) -> D_params=108M
    + 4919282
+ 1574: n928 (interpolate nearest in D). D_params=4M, small3
    + 5021656
+ Forgot to apply conv2 in 925
+ 1575: n929 = n928 + apply conv2
    + 5029326
+ 1576: n930: no strided conv
+ 1577: n931: no branch conv
+ 1578: n932: simpler G (2M)
+ 1579: n933: dilate
+ 1581: n935 (nproj=16)
    + Gets somewhere by @20,000
+ 1582: n936 (nproj=64)
    + Does GP converge faster?
+ 5069982 / 5075226: n937 (nearest in G)
    + Nice @20000…
        + The best @50,000, it seems. Could be a fluke but activated wires is closer to real distribution than nets with strided convs… Maybe we could maxpool instead of strided conv?
    + __5076480__: small4 data
+ 1583 / 5073601: n938 (strided convs in three branches, plus a concatenated branch.
    + Kay. slightly too many wires...
+ 1584 / 5074710: n939 (dropout in D)
    + Pretty good too. (like 937)
    + __5078470__: small4 data
        + Much faster than n937 (well, ~8 times fewer params)
+ 1585 / 5074955: n940 (dropout in G)
+ 1586 / 5075182: n941 (dropout in both)
+ 1587 / 5075183: n942 (one common branch conv)
    + 5085041: small4 data
+ 1588: n943 (n939 + maxpools instead of strides in D)
+ 1589: n944 (3 conv layers in common branch)
    + worse?
+ Ok. Use n937 but change how we sample interpolates for GP.
+ 1590: n937
    + 5099345
+ 1591: n939
    + 5099339 contd
        + Kay it works… __We'll use this WGAN-GP__
+ 1592: n945 (n939 + softmax)
    + Wire not diverse enough
+ 1593: altered
    + 5102176
+ 1594: n946 (n939 with more stuff in G)
    + 5104868 cont
    + 5123278 cont
        + very nice
    + 5135522 cont
+ 1595: more convs in D
    + 5112112
        + Big meh. Overfit?
+ 1596: same but n512=512 in G
    + 5112405
        + Not very bueno.
+ 1597: n949 residual connections in D
    + Explodes… Why?
+ 1598: n950 (corrections)
+ 1599: fewer convs in trunk
    + Ok better.
    + Ends up exploding…
+ 1600: n952 try again
    + Fat explosion
+ 1601: n953 (fewer params in common branch)
    + Explodes
+ 1602: n954 (replace nearest by linear)
    + Better
+ 1603: n955 (more G params)
    + 5123224
        + __Not good__
            + So common convolutions are not good for us… Not sure why, but ok.
+ 1604: n939, b_s=8, small4 data
    + meh
+ We clearly get better results when we don't have those common branch convs in D…
    + What if we add a conv to every branch?
+ 1605: n956 (939+ one more conv in each D branch)
    + 5126907
        + Pretty good @10,000
        + nice @35,000
+ 1606: n957 (and one more conv)
    + Seems fine
    + 5129115
        + Mhhhhh........…
+ 1607: n958 (one more conv in G)
    + 5135426
        + Nope. Not bueno at all.
+ 1608: n946, small4 data
    + 5136254
        + Not bad, but not perfect
+ 5136260: n946, full data
+ TODO: grid of 8 samples animated at once! To show diversity!
+ Are common branch convs really what's messing us up?
    + Or is it that we stop increasing feature maps?
    + Does it work better if we don't stride?
+ __Start from n946__
+ 1609: n959: three convs in common branch but they're not strided
    + 5146820 -> __Same as strided. Loss goes back down after @2000__
        + Ends up pretty terrible
+ 1610: n960: replace Conv1ds by DBlocks
    + 5146821
        + Seems nice @20,000
        + So it's the additional conv layers' fault…
        + What if we keep increasing the feature maps?
+ 1611: n961: correct DBlock in input layer in D p-branch
    + 5147358
        + Good. Seems slightly worse than 821 actually...
+ 1612: n962: change DBlocks to facilitate flow of input to output
    + Seems faster
    + 5147434
+ 1613: n963 = n959 with increasing fmaps (168M params...)
    + We can't train such a big model…
    + Looks pretty good @1000 tbh
+ Residual (addition) connections
+ 1614: n964 (3 additional res con in D common branch): 41M params
    + Looks explosive
    + 5148037
        + Massive explosions
+ 1615: Residual blocks like in resnet paper (1512)
    + 5148243
        + Doesn't look great
    + 5157410
        + Not good @200,000
+ 1616: n966: ResBlockDown
    + 5161289
        + Nope
+ 1617: n967: ResBlockUp in G
+ 1618: n968: minor corrections
    + 5162658: tiny data
        + Not good
    + 5162661: small4 data
        + Nope
+ 1619: n969: reduce D complexity (0.9M)
    + 5191233
        + Much better than above below, when reducing G complexity.
        + Difference with above is n64=16 instead of 64 and nproj=8 instead of 16
+ 1620: n970: reduce G complexity too (0.9M)
    + 5191301
        + Worse than above with G_params = 4M
+ Strange how we first have a sort of average between training samples and then the quality degrades…
+ 1621: n971: Fewer ResBlocks in D, only Downs
    + Very slow learning…
+ 1622: No cat branch, 
+ 1623: n973: No residuals in D
    + 5230314
+ 1624: n974: more complex G
    + 5230313
+ What if we have more filters in G rather than D?
    + Also it seems like D residuals are screwing us up, so maybe we can't use them at all.
+ 1625: n975: no common branch in G
+ 1626: n976: more fmaps in G
    + 5252061
    + 5265616 cont'd to compare with 5265605 (no dropout in D)
+ 1627: n977: no dropout in G
    + Maybe better @5000 than other two below
    + 5256598
        + Looks better than above.
            + __→ Don't use dropout in G__
+ 1628: n978: no dropout in D
    + 5256457
        + Looks very very close to above.
            + Maybe prevents the score from going back up a little bit?
                + __→ Use dropout in D only__
    + 5265605 contd
+ 1629: n979: no dropout in either
    + 5256717
        + Very clear that loss goes back up after ~20,000 epochs
+ 1630: n980: n979 + ResBlocks in G tips
    + 5262656
        + Nice.
+ 1631: more ResBlocks in G roots (6M params)
    + 5264082
        + Nice.
+ 1632: even more ResBlocks in G (6.5M params)
    + 5264109
        + Still nice
+ Which one's the best @50,000?
+ 
+ 1633: heavy 0.5 dropout after branch concatenation
    + 5269401
        + Pretty odd artifacts because of dropout, but ok @38,000
        + Pretty okay for a meme try
+ 1634: same with small4 data, batch_size=8
+ 1635: same with full data, batch_size=32
    + Mkay doesn't look very good @5000 but kind of expected, like 1634
+ 
+ Can we implement cross-validation?
    + Yup. train_wgangp_xval + dataset_one_xval
    + 1639: let's see how it looks (n983)
        + 5275888: 100,000e (check val_loss.png in directory)
+ Does QT make a difference?
    + Not sure. No qt makes the distributions absolutely spikey, so it's expected that the GAN can't really realise them.
+ Larger kernels in early layers.
+ n984: more reasonable dropout, large kernels
+ 1640 / 5277863: n984.
    + 5278491: n984, full dataset
+ 1641 / 5278419: n985: no common branch convs (<100k params)
    + Yeah that seems much better
    + 5278500: full dataset
        + Terrible.
+ 1643 / 5278693: n986: +three common branch convs (stride=1)
    + 5278860: full dataset
    + 1644: small4 data
+ 1645: n987: fewer params in G (1M) and D (0.3M)
    + 5292938
        + Val loss goes up after ~9000
        + Samples look poor
        + Actually better @50,000
+ 1646: n988: more params in G (1024, 19M)
    + Much much better…
    + 5293445: 50,000e, tiny data
    + 5293379: same with no dropout (n989)
+ So it's possible we need a phat generator.
+ 1647: n990: large k_s convs at tips of G
    + Definitely something in the wire distribution… or is there?
    + 5295488: moar
        + Val loss separates around 12000...
+ 1648: n991: resblocks (again...) in D
    + 5296883
+ 1649: large16 dataset (stride=1, 16 samples → 30,000 sequences)
+ __PANKAJ__
+ 1650: n980
    + 5304150
    + 5310062
        + I mean it's good but distance / time diff distribution is really really not there…
        + G:
            + No cat branch. Three ResBlockUps in common trunk. Then two in each branch and finish with a ResBlock and a  k_s=7 conv. __No Dropout__
            + n512 = 512. __n_params=X__
        + D:
            + No cat branch: 1 entry layer k_s=7, +2 strided convs in each branch, then two common branch convs and lin. __No Dropout__
            + nproj = 8, n64=16. __n_params=X__
+ 1651: n971
    + 5304053: okay @50
    + 5323471
    + 5385213
        + Activated wires aren't making it.
        + @400 Starting to look better, but loss is plateauing. Does it get good eventually?
+ 1653: n992: arch
    + 5306851
        + Wire dist no good
+ 1654: n993: fewer branch layers in D
    + 5309514
        + Worse.
+ 1655: n994: more params in G (2M)
    + 5309877
        + Overfits @10
+ 1656: n995: difference conv in D, tiny data
+ 1657: n996: no res blocks.
+ 1658
+ 1659: n998: more convs for diff
    + 5320466
        + Doesn't seem to overfit, this time
            + -> No common branch convs
+ 1660: n999: strides
+ 1661: n1000: more G params
+ 1662: n1001: more D params
+ 1663: n1002: disable difference convs
    + Why is this bad?
    + 5327463
        + Well it might be bad but it doesn't overfit…
        + And activated wires is about right…
        + 5327914 moar
            + It's actually pretty good @200,000
        + 5334407 moar
+ 1664: n1003 = n982
+ 1665: n1004 = n980 + two ResBlocks in w branch G
    + 5323607
+ 1666: n1005 = n1002 with same G as n980
    + 5323628
+ 1667: n1006 = ^ + n64 = 16 in D rather than 64
+ 1668: n1007 = ^ + two common trunk strided convs in D
    + Much better. So in D, having common trunk convs definitely helps…
    + How many is too little? How many is too much?
+ 1669: n1008 = ^ + reenable difference convs
+ _Try square difference convs?_
+ 1670: n1009 = ^ + square difference convs
    + 5327458
        + Not good enough. + Clear overfitting curve
+ 1671: n1010 = n1007 + one fewer common branch convs
+ 1672: n1011 = n1007 + one more common branch conv
    + I mean it looks better but it also looks like it overfits more from val loss curve.
+ 1673: n1012 = n1011 but fmaps are capped at 512 in D
+ 1674: n1013 = n1011 + difference convs
+ 1675: n1014 = more convs in G
+ 1676: n1015,  geom_dim=2
+ There's a massive difference between common trunk convs and not in D…
+ It's maybe the most apparent cause of overfit…
    + n1002, despite having n64=64, doesn't overfit @50,000 unlike n1007
    + 1677: n1016 = n1002 + n64=16, dropout before lin0
    + 1678: n1017 = ^ + one normal conv k_s=1 in common branch
        + still ok @5000
        + Starting to have a stroke @10000
        + It's a big explosion. Not sure why tho
    + 1679: n1018: n64=128, 9M params, bias=False in convw0
        + Now it works… @4000
        + So when D is capacity limited, we get loss explosions…
            + 5329558
            + Interesting but explosive loss curve.
    + 1680: n1019: no branches in D, n64=64
        + Actually looks decent, but explosion inbound?
        + 5329794
    + 1681: n1020: more convs in D
        + Seems to diverge @10,000
        + Explodes
            + What if no dropout?
    + 1682: n1021 = n1019 + no dropout
        + 5330478
            + nice
                + Distance distribution still struggles @200,000
        + 5334270 full data
        + 5334424 moar
            + This is __ALMOST__ on point. I think we just need a little bit more long term vision.
            + Validation curve looks amazing. Distance comp and activated wires still a bit off.
            + __Discriminator is simple__
                + No branches, wire embedding then conv(7, 1), conv(3, 2), conv(3, 1), conv(3,1) → Linear
                + Four conv layers in total, with fmaps doubling.
                + n_params = 350k
    + 1683: n1023: n1020 + no dropout
        + 5330491
        + 5334533 moar
            + Overfits. Distance is okay but not perfect, activated wires is too high
    + 1684: n1022: diff conv
        + 5330603 → best
        + 5336786 moar
            + Pretty good actually: D n_params=350k, g n_params = 4.6M
    + 1685: n1025: diff squared conv
        + 5333729
    + 1686: n1026: more convs in top of D
        + 5334208
    + Notes: I've missed a day of sleep.
    + __Perhaps the bad distance isn't due to the wrong information being passed, but a lack of long-term remembering...__
        + Damn. That might just be it. Compare 1687 with 1688`
        + I need a better control. n1028
            + 1688 / 5348439: control for 1687
                + It's good tho...
    + 1687 / 5348440: n1027 no difference conv, first conv is k_s=33
    + 
    + What about a purely dilate D?
    + 1689: n1029: w=3, 5 layers
        + 5352157
        + 5390059 contd
    + 1690: n1030: 7 layers → Doesn't look too good @5000 Instant overfit and distance is worse than simpler models.
        + 5357330
            + Not good. @50,000 some interesting features. activated wires?
            + 5376185 moar. __Is this the best we ever got?__
            + 5385902 moar. __Das it mane.__
    + 1691: n1031: 2 layers, n64=256 (n_params=450k)
        + 5358404
    + 1692: n1032: dilate goes up then back down in D
        + 5363968
    + 1693: residuals, only two stride convs
        + 5369646 / 
    + 1694: n1028 + no branch upsampling in G
        + Looks ok actually
        + 5378201
    + 1695: n1035: __Only__ branch upsampling in G
        + 5378191
            + Pretty good
        + 5384458 contd
            + Meh @200,000. Activated wires still too high
    + I think only branch upsampling looks better in the end
    + Neither works perfectly. **WHAT**
    + 1696: n1036: dilate in G tip
        + 5383363 → Pretty smooth D curve
    + 1697: n1037: large kernels in D
        + 5384465
            + Very meh
    + 1699: n1039: independent projection for w and p branches in G
        + 5385406
            + meh
    + 1700: n1040: similar D as n1030, n64=16 but more increase along depth
        + 5391970 → Struggles to learn distributions, but activated wires are able to go down.
    + 1701: n1041: fewer params (n64=16 and only increase marginally every time)
        + 5391318
            + Okay, not as good as 1030 it seems...
    + 5392942: n1030, full dataset, 100,000e
        + Meh. @80,000 D loss doesn't go back up like for smaller datasets…
        + Let's go on and see
        + 5512976 contd
            + Act wires seems to be moving down @160,000
            + Pretty clean but slow to learn. @200,000 most samples still look like blurry circles.
    + 5392977: n1030, large16 dataset, 1000e
        + Meh
            + Bit of a struggle @270, but might get there eventually. 
    + 1703: n1043: 7 dil convs, double fmaps, cap at 512
        + 5429343 → looks better than below early on
        + Gorgest @70,000
        + Insanely good
        + 5459748: full dataset
            + Very slow.
                + @92,000 activated wires starting to shift toward real distribution
                + 5599364 contd
                + 5660536 contd
                + 5718909 contd
                + Looking better and better @220,000
                + @250,000: Looks fine. Good diversity, good single samples, distributions aren't perfect but can probably train for longer and then pick a good one. It looks like high-activated samples don't go away even at this stage. Not sure if they ever do.
                + __Could continue training after conf__
        + 5459783: large16 dataset
            + Pretty poor @150
                + Quite meh
    + 1704: n1044: fewer convs, more dilation
        + 5429344
    + 1705: n1045: one res connection in bottom of D
        + 5431464
            + Pretty good @40,000
            + __Insane @100,000__
    + 1706: n1046: 7 dil convs but fmaps don't increase
        + Doesn't look great @5000
        + 5434888
            + Pretty good @100,000 tho, for a disc with only fmaps=64 convs
    + 1707: n1047: 3 dil convs, increasing fmaps. 38k params
        + Figured out activated wires impressively quick (10,000). However, feature distributions aren't the best
        + 5434864 → Distributions suffer from lack of params, but overfit disappears completely. __Activated wires distribution looks amazing. We might have done it.__
        + 5448675: full data, batch_size=16
            + @100,000 I can see some patterns, but activated wires is way up still.
    + 1708: n1048 = n1043 + dilate in G as well
        + 5447606
            + Very good.
        + 5465633: more
            + Very nice. Very nice indeed.
        + 
    + 1709: n1049 = n1047 + n64=64
        + 5451663
            + Actually still not good enough it seems.
    + So most of our dilate nets seem to perform well after a certain number of iterations, on the tiny dataset. We have yet to see the same nets do well on the larger datasets. Can we try them on intermediate sizes too?
    + Let's try 1045 on small3 and small4
    + 5465609: n1045, small4 dataset
        + Good
        + __Insane__ @130,000. Good.
        + 5544495: contd
            + Good. Some samples still noise @300,000 tho
    + 5465610: n1045, small3 dataset
        + Nice
        + 5505885 contd
            + Good good good
        + 5544541 contd
            + Learned the training set perfectly it seems. Let's stop here
    + 1710: n1050: 1049 + pass n_wires to networks at initialisation. Remove VAE
        + Works. Can we do a job or two with part of the real dataset?
    + 1711: n1051: 1045 + another residual connection in D (total 2)
        + dataset_real_8
        + 5506757
            + Nice ish
        + 5544640 contd
            + So it's good but it's really just pretty much copying samples from the training set. Can we expect any better from a GAN?
                + Fix animate_time.py for this job
        + 5583058 anotha one
            + → Pretty good. Can it generalise to larger dataset, rather than overfitting to few samples?
    + 1712: n1052: Residual Dilated Convs, dilation=2^L, 11 layers.
        + 5564777: dataset_real_8
            + Good distributions @60,000. Samples are appearing. But they're very similar to the training ones. Maybe D's receptive field shouldn't be as big, or we should lower its parameters to avoid the overfit we see here.
            + 5640726 contd
                + Very very slow, results aren't that good honestly.
    + 1713: n1053: half params in both D and G. Reduce D depth by 3 layers.
    + 5628816: full real data, n1053, __latent_dims=500__
    + 1714 / 5679223 / 5697937: n1045, (only one res) real data_4
    + 1715 / 5681816: n1048 (dilate in G too), real data_4
        + Better distributions than 1714
    + I'm afraid that res connections are making some samples noisy as a result. Can we test res vs no res?
        + Well 1715 (n1048) has no residual, but it's not identical to 1045.
        + 1716 / 5681693: n1043 (like n1045 with no res), real data_4
    + 1717 / 5687945: n1045, dataset_real_16  → seems nice @100,000
        + 5747555 contd
            + Pretty good. Another IOP candidate?
+ We're continuing 1714 to see if the same loss patterns as 1715, 1715 appear eventually. Not sure what the patterns mean (two peaks in G loss) but it might be something worth noting…
    + @150,000 still no rise in G loss. Does this have any importance?
+ 5704409 / 5780979: n1043, dataset_real_64, batch=8
+ 5704410 / 5784214: n1045, dataset_real_64, batch=8
+ 5704411 / 5780995: n1048, dataset_real_64, batch=8
    + echt
+ 1719 / 5723034: n1045, latent_dims=10, dataset_real_16
    + Bit disappointing @40,000
    + Fine @100,000
    + 5784204 contd
+ 1720 / 5723024: n1043, latent_dims=10, dataset_real_8
    + Cannot really deduce what reducing latent_dims does from this...
+ 5764993: n1054: dilation goes up and down in G. real_dataset_16
    + Quite nice actually, @40,000
    + Activated wires is better than e.g. 1045, quite early on.
+ /!/ Cannot show latent space interpolation animation, it just outlines how bad the model is at diversity…
    + It's very interesting tho.
    + 1722: smallest networks ever. 2 latent dims. n64=4.
    + TODO: __find a net architecture that is creative__
    + 5825607: n1056 (lower n params, 2 residuals in D, l_d=100)
    + 1725 / 5871419: 16 samples, n1057
        + Overfit appears in val loss @15000
    + 1726: 4 samples, n1057
    + 1727 / 5870994 / 5925708: n1058: simplest D ever, 4 samples
        + Actually able to do it… With a few thousand params…
            + Sucks @400,000
            + Overfit. Latent space non-smooth. Dataset probably too small to do anything.
    + 5908605: n1058: full real dataset
        + Validation loss looks better @5000…
            + No overfit yet. __200k generator iterations__
            + @400k gen its, validation loss still good. 
    + 1728: n1058, real_dataset_s64
    + 1729 / 5947393: n1059: one more layer in D
        + Honestly, it gets better and the validation loss looks pretty good @400. Check latent space?
    + 5948835: n1060: no res in G. Simple stuff. BatchNorm.
        + Val loss looking great @26000 generator iterations (@480 epochs)
        + Samples looking pretty meh tho.
        + Latent space is meh as well.
        + Hmm kay. Looks like it improves?
        + Let's go for more: 
        + 6003452: contd
            + Looks nice. Act. wires still hanging around 1000/sample at @1000e
    + 6003443: n1061 (three more convs in D), s64.
        + No overfit @50.
            + Nice-ish interpolation @850 (50,000 generator iterations)
            + We clearly see a change in wire positions of clusters as we interpolate, which is nice.
    + 6042908: n1062
        + Hmm. Wire activation is going down it seems… @600
    + 1733 / 6095288: n1063: 3 geom dims (remove theta)
        + Looking fine @300. though clearly resolution is meh.
        + Change strided convs to something better…
        + 6143020 contd
        + 6187563 contd
            + Interpolation sucks? Or really subtle. Maybe just more subtle than earlier in training.
    + 1734 / 6126523: n1064: dilated convs to replace strides. One more layer in D.
        + Still nice-ish latent space @600
    + 1735 / 6165990 / 6170723 / 6220993: n1064 control with real_dataset (no stride,300samples)
    + 1736 / 6166111 / 6172601 / 6221036: n1066 = n1064 + one more branch layer in G. __Actually I messed up. The extra layer isn't used.__
    + 6182486 / 6185883 / 6225097/6291075: n1068 (more balanced params in G)
        + Samples seemingly look better, visually, @10000.
        + Nice @20,000. Act wires still a bit high. (40k gen its). Interp looks okay
    + 1740 / 6225859/6257735/6291400: n1070 = 1068 + check what happens with shorter branches in G
        + Not looking too good @2300…
        + Okay @10,000
    + 1741/6238046/6250602/6262508/6292976: n1071 = n1068 + one more branch conv in G
        + Looks worse than previous (n1070)
    + 1742/6258030/6291431: n1072 = 1071 + cat branch in G
        + Hard to say @5000
    + 1743/6356215/6395035/6425375/6473500/6532027: n1073: conv latent space then upsample in branches.
        + Looks like it's overfitting @30,000
            + Overfits @30k
        + 1744/6356616/6395036/6410499/6428986/6460919/6473464/6510538/6586401/6635258: n1074: Strided convs in D. n_params well balanced. Faster by about 30%.
        + Distrubutions looking bad @10,000
        + Looking hella strong @40,000. No overfit. Distributions improving. Act wires improving (slowly).
        + Weird spike @45,000. 
        + Recovers later on. Not sure if it'll ever get any better tbh.
        + Gets pretty good eventually @100k…
        + Hard to say again if it'll get any better. Losses are flat.
    + 1745/6411083: more convs in D.
        + Divergence in val loss @12,000… D loss also goes down at the same time
            + Can we fix this by increasing G params?
    + 1746/6439534: double n512 in G (256)
        + Still overfit. This time val loss splits around @6000, so even earlier…
        + Bad overfit. No bueno.
    + So maybe the solution is fewer params but more convs? Or something…
    + Clearly by going over 1M params in D, we have caused overfit to occur, since previously e.g. in n1074 (600k params), we don't see val loss split off even after 30,000 epochs.
    + 1747/6463844/6516146/6570961: n1077 = n1074 + residual connections
        + No overfit yet @20,000
        + Nice latent space @40k. Too many act wires still
    + 1748/6463854/6516153/6570963:n1078 = n1077 + one more conv in resblocks
        + Not that much slower than parent…
            + Weird spike just before @40k… What happens next?
            + Hard to tell wtf is going on. Act wires is okayish, but samples look too noisy still.
    + 1749/6546700/6587027/6639727/6691210: n1079: simplest D. Idea is to test how results are affected by changes in D.
        + Somehow it's good… I don't understand any of this
    + 1750/6611655: n1080: no convs. Just 2 lins
        + Pretty bad, somehow overfits
    + 6611665: n1081: same but fewer D params
        + Overfit? Really?
    + 6617299/6635189/6668702/6703092: n1082 = n1079 + simpler G
        + Works but samples lack detail.
    + 6692740: n1083 = 1082 + only one conv (w/p) in G
        + Clearly the generator can only make gaussian-like data…
    + 6702111/6724691: n1084 = n1083 + 6 conv layers (64fmaps)
    + n1085: try no linear layer in G. Only ConvTs. Seems meh
    + 6723972: n1085: instead, project to smaller seq initially, like 4
        + Still seems to lack detail...
    + 6727654: n1086 decreasing fmaps in G
        + Same. We only get gaussian like data.
    + 6763779: n1087: strides in D
        + big ehrt. Maybe we can't have this few params…
    + 6772349: n1088 = n1086 + more params in G
        + Much better. Note that we didn't improve Disc at all here.
        + Best between low params and high params it seems. Interp is meh tho
    + 6790651: n1089 = n1088 + yet more params in G
        + For some reason the distributions look way more spotty than with fewer params… Looks like the generator is able to generate the same sample every time.
    + 6821155: n1090 = n1089 + circle track progression feature (train_wgangp_circle, evalf3, dataset_real_circle)
        + Converges quickly but result sucks and low diversity.
    + 6862009: n1091 = n1088 + strides in D + circle track
        + Overfit @17500. Despite having <500k params...
    + 6901661: n1092: only stride 2 convs in D, slightly more fmaps
    + 6929094: n1093: LayerNorm? Must use elementwise_affine=False otherwise we get unstable.
        + Doesn't look too good tbh.
    + 6961084: n1094: fewer strides in D, fewer fmaps.
        + Meh
    + 6978009: n1095: difference fmap distribution in G. Put LN before act.
    + I should probably use largest dataset possible and reduce the number of epochs + increase the batch size.
    + 1766/6984136/7059958: n1096: max net, max data. Conv all the way up/down to len=4. No LN.
        + Not bad tbh.
        + Still overfit @30 epochs.
    + Try branch in G
    + 7027953/7064154/7138112: n1097 = n1092 + two layers in w and p branch. Small data.
        + Yeah already much better I think
        + Quite nice. So much better than with no branches...
    + 7041399/7064115: n1098: slightly more params in G (G 700k, D 500k)
        + Looks quite good in interpolation too :)
    + 7041409: n1099: replace strides by dilate in D
        + No much difference. Quite a bit slower.
        + Evidence of overfit @10k
    + 7060363: n1100 = n1097 p branch depends on w branch tip
        + Doesn't look too good
    + 7072984: n1101 = same with residual connection
        + Meh as well
    + Best so far seems to be n1097. What if we branch more?
    + 7083673: n1102 = n1097 + make one more layer into branch
        + Doesn't look that much better than 1097 tbh.
    + 7109271: n1103 = n1097 + wire prog comes from w branch
        + Wire dist goes Gaussian. Yup clearly no bueno.
    + __Data parallel?__
    + 7170451: n1104 = n1097 + Double layer in G, 4-parallel GPUs
    + 7174631/7174781/7191227/7191346: n1104, test of DistributedDataParallel
        + First three failed because of serialisation problems.
    + 7191350/7207827/7218648: For real now (n1104)
        + Seems worse than 1097…
        + Same weird loss curve bump as control n1097
    + 7207159/7213365/7219535: control (n1097) + mistake no activation
        + Weird loss curve bump @20k.
    + 7218937/7252756/7302817: n1105 (activation after Double)
    + 7219964/7252755: n1106 (more fmaps in G, Double layers in P branch)\
        + Not bad @10k.
    + 7265422/7310070/7342376: n1107: different G: 128 fmaps everywhere except last layer
        + Interesting @50k.
    + 7368884: n1114: ^ + nproj=16
        + Not too different from nproj=4
    + 7369155: n1115: ^ + no projection, one-hot wire straight into conv like other features. Very slow
        + No significant difference in how wire gets handled it seems.
        + It's pretty good tho. I like that latent space interp.
    + 7327965: n1108: Linear in D only
        + Overfits quickly and results suck
    + 7328240: n1109: Linear in G too
    + 7328452: n1110: more params in G (1M)
        + Gets wire/layer distribution fairly well. Other features and sequence not that great tho.
    + 7343165: n1112: Even more params in G
        + Overfits fast
    + 7343164: n1111: Linear in G only
        + No overfit, but doesn't really work. Probably lacking params in G
    + 7368325: n1113 = ^ + more params in G
        + Better distribution of features but still gaussian wire
    + 7404519: n1116 = ^ + one branch conv
        + Ok, instantly better wire relationships
    + Maybe try shorter sequence, only wire feature
    + 7421144: n1117: full convolutional generator (no lin)
        + Meh. Samples very similar.
    + 7421224: n1118: ^ + batchnorm in G
        + Pretty bad. Wire looks blurry.
    + 7445892: n1119: ^ + similar D (128 fmaps max, 9 s=2 convs)
            + overfits
        + 7482786: n1121: ^ + less params
            + meh results
    + 7460269: n1120: ^ + only three convs in D, no stride
    + TODO: chunk stride does not provide true data augmentation from the point of view of the discriminator's convolutional layers. We need to implement track permutation for better results.
    + 7529734: n1121, augmented data
    + 7529767: n1119, augmented data
    + 7556547: n1122: branches in G
    + 7570292/7636935: n1124: ^ + more fmaps in D (__params equal to G__)
        + Clean distributions. Samples not so much. Can keep training
        + Looks cool @20k. Distributions ok, patches still too extended...
    + 7562288: n1123: longer branches in G
    + 7588754: n1125: more fmaps in both ()
        + No apparent difference
    + 7591515/7657781/7713345: n1126: residual in G
        + Act wires moving toward low side @10k
    + 7601884/7657691: n1127: residual in D
        + Some sample start getting right number of act wires @10k. Keep going
            + Nah still not great after more training.
    + 7626747/7635009/7657669: n1130 = ^ + sequence_length=256
        + Still lacks tight clusters @20k. Keep going.
    + 7676286: n1134 = ^ + nproj=256
        + Nope. Doesn't improve.
    + 7635080: n1132: ^ + dilations in D instead of strides
    + 7627029/7674647/7755309: n1131= n1127 + double-conv residual layers. Big boi model
        + Trouble getting any better. Pretty sure peak performance is reached around 15k "epochs" (20k gen its).
    + 7624077: n1128: n1122 + larger kernel in D tip
        + No huge difference.
    + 7624962: n1129: n1122 + replace BN by InstanceNorm
        + No difference.
    + n1135 = n1131 + geom_dim=2 + n_features=3 (no prog)
    + 7783417
+ Add occupancy to disc maybe?
+ 7850742: n1136: occupancy in D (see printouts)
    + Actually makes a difference… But the samples don't look super good
    + Yeah it works to regularise the activated wire distribution. But samples still look pretty bad.
+ 8098686: n1137: ^ + lin0 in G
+ 8107883: 1809: n1138: feature extraction from source features by dilated convs.
+ ^ Both look less stable than pure conv one
+ 8147078/8185920: n1139: only project with linear layer, no act or norm
    + Ok actually. Structure appears @10k
+ 8147670: n1140: simplest G (no activation)
    + Spikey. Samples look poor.
+ 8186399/8228985/8315134: n1141: more layers, 128 fmaps. This gon be slow
+ 8186548/8228986/8315135: n1142: ^ + norm before act in G
+ 8186612/8229022/8315137: n1143: ^ + linear interp mode in G
+ 8314748: n1144: n1141 +  nproj=16
+ 8439312: n1145: no wire embedding in D
    + Nope. Unstable.
+ Okay so more layers doesn't seem to improve our sample quality. What's the solution then?
+ 8533696: n1146 = n1143+ only 4 layers in both nets. G has 3 in branch
+ 8534120: n1147 = n1146 + no branch in G
+ 1814/8600630: n1147 + pca transform instead of QT
+ 8601444: n1148: nproj=16, conditional BN on occupancy for W branch in D.
    + Explodes.
+ 8606639/8694453/8746784/9039771/9574667: ^ + weight init for D's linears (to avoid explosions early on) n1148
    + Best so far. Conditional BN seems to induce weird W distribution.
+ 8747568/9526378: n1149 = n1146 + no batchnorm in G (+ weight init)
    + 8772604: ^ + no Linear weight init (only conv)
        + Looks similar.
+ 8747569: n1150 = n1146 + BatchNorm(affine=False) (+ weight init)
+ 8775844: n1151 = ^ + no linear, only convTs in G. Removed output_padding as well
+ 
+ 9695530/9707290: n1152: sequence_length=256, occupancy in later layers in D, no conditional batchnorm but just repeating of occupancy.
+ 9702143/9707484: n1153: ^ + large kernel first layer in D
    + Same issue of too noisy hit patches.
+ 9720040: n1154: ^ + more layers in p branch (G)
    + Looks like similar problem to above.
+ 
+ 9720255: n1155: linears and LayerNorm
    + Wire distribution seems fine in fact. Because of occupancy removed?
    + Diverges badly and then distributions are off.
+ 9724946/9741035: n1156: ^ + elementwise_affine=False in D's LayerNorms
    + Much better stability than elementwise_affine=True
    + Sample quality low, distributions imperfect (too central)
+ 9724989/9742467/9815383/9830567: n1157: ^ + occupancy
+ 9815075: n1157 + dataset_augment3 = limit number of combinations by resetting the seed of the RNG.
    + Not looking good 
+ 9815083: n1158 = ^ + no layernorm in D. It just seems to make the distributions worse.
+ 9815143: n1158, sequence_length=16, augment2
    + Looking unstable af
+ 9815148: n1157, sequence_length=16, augment2
    + Better in stability
+ 9820728: n1156, seq_len=16, augment2
+ 9820731: n1156, seq_len=16, augment4 = 10,000 samples
+ 9836519/9849229: n1159: n1156 + no xy
+ 9848038: n1160: ^+ tanh after p projection, more params in G before last layer
+ 9849202: n1161: convolutional with Gaussian weight init
+ 9871638: n1163: ^+ nproj=64
+ 9873276/9890078: n1164: n1161 + w branch has 3 convs (G)
    + Not seeing too much diff @40k
    + No improvement.
+ 9860721: n1162: ^+ flatten instead of mean in D
    + no diff
+ 9883011: n1165: n1159 + 3 convs in w branch with upgoing params
    + Looks identical to 1159.
+ 9890077: n1166: ^+ embedding has 3 layers in D
+ 
+ Repeat experiments with dataset_augment4 (10k samples)
+ 9899870/9909829: n1159 (lin), ld=100
    + Occupancy doesnt go down. D loss very flat.
+ 9899874/9910056: n1161 (conv), ld=100
    + Same. Distributions worse than with n1159.
+ 9917454: n1166, ld=100
+ 9918068: n1167=n1161+occupancy, ld=100
    + Gen loss goes up, as if it can't control occupancy whatsoever
+ 9919215: n1168: n1159+ reduce gumbel temperature to 1e-3 over 200k gen forwards
+ 9923183: n1169=n1159+occupancy
    + Gen loss goes up too, like n1167
+ 9926831: n1170=n1167+branches in G
+ 9934559: n1171=^+ one strided conv in G for seq_len=32
+ 9935476; n1172=n1161+strided convs in D
    + Doesn't work even at 2900 epochs…
    + We just can't get the model to learn that we want it to hit a limited number of wires in each sample...
+ 9943754: n1173: no proj layer in D, more params in w branch in G
+ 9945793: n1174: 2 more layers in D, no normalization
+ 10008046: n1175: n1168 + occupancy rescale
    + Doesn't fix @400e
+ 10034694: n1176 = n1167 + occupancy rescale
+ 10053723: n1177 = ^+ linear first layer in G
+ 10078331: n1178=^+ no layernorm in D
+ NONE OF THESE WORK FFS. How do we 
+ 
+ 10103265: n1179: G has branches from the projection layer down (independent upsampling)
    + Wait now it seems to make a difference.......??? Well we have one wire that's over activated so.
    + The only one where activated wires seems to really go down as a function of training epochs.
+ 10104335: n1180: D has branches too (1 layer)
+ 
+ 10161464: n1181: single layer in G and D
    + Bad.
+ 10166193/10172192: n1183 = ^+ singlegp (train_wgangp_augment2_weightinit_noxy_singlegp)
+ 10162429: n1182: two layers in G and D
+ 10182705: n1185 = ^+ singlegp. Just want to check if it works as well as two gps.
    + Huh so we get rid of the GP spike at epoch 0…
    + Overfit. Large number of params (10M), so that makes sense
+ 10173357: n1184: hard=False and anneal gumbel tau to 1e-3
    + Nope.
    + 10194721: correction (wasn't incrementing gen_it)
+ ~~10194753~~: n1186~=1184+1185 + log before gumbel
    + 10196416: Fixed to logsigmoid before gumbel
+ 10197070: n1187: logsoftmax before gumbel
+ 10201060: n1188: conv D with maxpools
+ 
+ Let's run some tests from the n1179 baseline:
    + 10240281: n1189 = n1179 + no occupancy
        + **ACT WIRES DOES NOT GO DOWN** (as fast)
    + 10240383/10405651: n1190 = n1179 + singlegp
        + Still works!
        + Interesting loss curve. Act wires seem to be close to matching the distribution eventually (@10k)
        + Distance distribution is not well matched. Probably because we don't give D any geometrical information.
        + **Looks the best out of all experiments for the past few weeks...**
    + 10240436: n1191: no lin0 in G, only convs from latent space to out in both branches
        + Works somewhat ok
        + Doesn't seem as good as n1190 @10k.
+ **Embedding**: train_embedding.py, cdc_embedding.py, train_wgangp_embedding.py
    + n1192: 128d embedding, context=6
    + 11435249: n1193, cdc_embedding9.10k.pt
    + Try max_norm=1.0 for GAN's training performance. As of now we're getting large loss values so it might be the problem.
    + We're seeing an imbalance in the wire distribution early on. Can we correct that with weighted cross-entropy in the embedding training?
    + embedding_17: try a better split between train and test: split wires rather than z slices.
        + Actually this is stupid. We need to make all the wires (vocab) present in the dataset, but the test contexts should be unseen during training.
        + So we need to pre-compute the training and test contexts.
    + 18: don't pick z at random
    + TODO: try picking the context semi-randomly, ie take a random selection out of the topk closest wires.
    + 11499576/11499608: test embedding training job
        + Works!
    + 1865/:e16, n1194
    + 11690939/11767798: embedding_23 (16d), n1194
        + Something happened… Can we do better with more emb dims? or fewer?
    + e24: 32d
    + e25: 64d
    + e26: 128d
        + Test acc worse (0.4)
    + So there seems to be a sweet spot…
    + It's 32 or something
        + Let's do a job with embedding_24 (32d)
        + 11767753: e24 (32d)
        + 11767756: e25 (64d)
        + 11767759: e26 (128d)
        + Okay they all seem to give worse GAN results than e23 (16d), so clearly higher dim is worse for the GAN. Let's try fewer dims.
    + e27: 8d
    + Embedding doesn't work. It fails to capture spatial relations somehow (hit )
    + n1195=n1045 + 2 geom dim
    + 1869/11854884 = weight init version of 1868 (n1196)
        + Doesn't make any diff?
    + 1870 = softmax
        + Looks okay tbh. I'll continue this. I want to try shorter seq as well.
    + 1871: n1198: removed 2 dilated layers from D. SL=512
        + 1872: just to check the validation loss at the start
            + Uhhh it's not good…
            + For some reason loss goes other way compared to 1871???
        + 1873/11877652/11920060: n1199 fewer G params
            + Same. Test and Train losses split early on…
        + 11920319: same with validation correction
            + muuuuuuuuuuch better.
            + Looking perfectly like WGANGP paper. **Great progress.**
    + Careful, we don't weight init the linear layer in G at the moment. Might be worth changing later.
    + 1874/11877485/11933038: n1199, dataset_augment5 (100k combinations)
        + Much better latent interpolation @100e than 652. 120k gen its.
    + 11920321: same with validation correction
    + Things I want to try:
        1. No one-hot wire in disc
        2. Different geometry representation. At the moment the generator produces more hits on the outer layers.
    + **found the mistake in validation!**
        + Concatenation has wrong order for xy and w
        + 
    + 11928367: n1200 / train_wgangp_postemb_now: no one-hot wire in disc
        + Wire diversity is worse than with one-hot wire...
    + 11930753: n1199 / train_wgangp_postemb_sphere: wire xy is in range [0, 1] rather than [-1, 1].
    + 1877: n1199 / train_wgangp_postemb_init: no weight initialisation
        + Wire dist looks the same early on.
    + 1878/11969019: n1201: strides instead of dilate in D
        + Looks like it's progressing nicely…
        + Looks quite good @160. We're starting to see more wire diversity. However the lowest activated-wire events are not really present....
    + Ah I should be careful continuing jobs. I re-generate the seed everytime and I should probably not do that. It means our test loss might be wrong.…
        + Let's try 1201 again without a job continuation. I want to make sure this validation loss is legit.
        + 12064805/12149704: n1201, same as 9019 but no use of initial_seed.
            + Looks legit @200. Going again.
        + 12188802/12394003: n1202: ResBlockDown in D
            + Still getting the excess toward the outer layers.
        + **Is the excess of hits in outer layers due to the architecture, the position representation, or something else?**
        + 12394861: n1203: n512=256 in G (same as D) to balance out params
            + Doesn't seem to help excess @10, @200.
        + 12394859: n1204: Instead of half-branches, full-branches in G (only lin in common)
            + Distributions look worse @ same number of epochs as 1202
        + 12740564: n1205 = n1203 + last two convs in G are k_s=33 followed by k_s=1
        + **It doesn't seem to come from architecture, at least not the things we changed above**.
        + Position representation? Run some tests.
        + 13249825: n1203, no wire rescaling (train_wgangp_postemb_geomtest.py)
            + Quite catastrophic… The magnitude of these vectors matters a LOT for the whole GAN training.
        + 13430041: n1206 = n1203 +  Lower LeakyReLU alpha to 0.02
        + 13430042: n1207 = n1203 + don't add but concatenate in ResBlocks
        + What about softmax but no xy in D? Does that fix the issue of excess in outer?
        + 13436240: n1208: no xy (train_wgangp_postemb_noxy)
            + Okay so it's definitely xy that's causing the excess, it seems.
            + Adding xy makes the latent space properties much better defined geometrically.
            + But it makes the W distribution much better.
        + 13546339: n1203, inverted wire definitions (i.e. wire 0 is now at the position of wire 4985, 1 is 4984, etc.)
            + **As predicted, excess is inverted. Now is in inner layers**
        + Two possibilities: custom geometry representation, like I did before,
        + Or check that removing xy from the gradient penalty fixes this, and does not change sample quality.
        + 13557641: n1209=n1203, cylinder of height=1, geom_dim=3
            + Quality of P distributions is meh, but W distribution is definitely better balanced.
        + 1887: train_wgangp_postemb_geomtest2, no cylinder standardisation. All wires have same norm
        + 13557646: wire cylinder like original wgpt26. (norm=1 everywhere)
            + All distributions good, but occupancy sucks.
        + 
        + **TODO: set up training script for no xy gradient pen**
        + 
        + train_wgangp_postemb_geomtest3 + n1210 = pass wire_sphere to disc and let it compute it. → No grad pen wrt xy.
        + 13573249: ^.
            + Okay so pretty much perfect wire distribution @60. But geometrical understanding is a lot worse it seems…
                + Not perfect: we have slight excess in inner and outer layers.
            + Wire occupancy is worse than others as well.
            + Interpolation looks good tbh, it's just that the occupancy is meh.
        + 
        + train_wgangp_postemb_geomtest3_ttur = ^ + two-timescale update rate (balanced updates (n_critic=1), lr of D four times that of G)
        + 13639996: ^, n1210
            + → Unstable
        + 13731820: betas=(0.0, 0.9)
            + → Still unstable.
        + 
        + **TODO:
        + n1211 = n1209 + SelfAttn + geomtest2
        + 1889: only attn in G → OK
        + 1890: One attn layer in G (both branches) and D → explosion
        + 1891: try again with one extra conv after attn in D
            + Still unstable grad pen…
            + Ah, perhaps it's the initialisation. No it's Conv layers so no initialisation.
        + 1893: Move attn to after first res layer.
            + → Unstable.
        + 1894: less clutter in SelfAttn class. And fix softmax dimension.
            + → Works now. Was it softmax?
                + 1895: Don't specify dim in softmax
                    + Yep. It was the softmax dimension. Thank god I found it.
        + 1898: n1211b (two attn layers)
        + ===
        + === **geomtest2: sphere surface with close point distance, like wgpt26
        + 13760939: n1211: one attn layer
            + Unstable.
        + 13760954: n1211b: two attn layers
            + Destabilizes @100.
        + 
        + === **geomtest3: no grad pen wrt xy. disc is given position tensor.**
        + n1212 = n1210 + SelfAttn + geomtest3
        + 1892: only attn in G
            + seems fine.jpg
        + 1896: attn in both with softmax fix
            + yes sir.
            + New version: with activation after SelfAttn
                + Looks similar. I'll drop the activation for now.
        + 13760841: n1212, geomtest3, one attn layer
            + Unstable
        + 13760861: n1212b: ^, two attn layers
            + Still stable…
            + Destabilizes @180
        + Perhaps it is unstable because, once again, of the learnable parameter (like in spectral norm)
        + 14403858: n1213: no learnable gamma
            + Doesn't look too good.
            + Way unstable @80
        + 14540285: n1214: disc attention goes before last resblock (D)
        + also, check out attention-augmented convolution maybe
        + 14612181: n1215: no bottleneck in SelfAttn
        + ~~14613165~~14814740: n1216: ^ + two attn layers in G, three in D
            + Unstable. So it seems this attention layer sucks...
        + 15233296/15700746: n1217 = n1212 (geomtest3) + no bottleneck + attn before last resblock + sigmoid gamma
            + geom_dim = 2, wire sphere is flat.
            + Distributions look neat af @90. Perhaps we could increase the attention in this one or something…
            + **Best candidate ever I think**
            + Stable and seems to make good progress over time.
            + Sigmoid gamma good?
            + What if we switch geom representation?
        + 15437413/15463451/15700736: n1218 = n1211 (geomtest2) + no bottleneck + attn before last resblock + sigmoid gamma
            +  = ^ + grad wrt xy + geom_dim=3, wire cylinder with norm 1
        +  ~~n1220: correct geom_dim in n1218~~
        +  geomtest4 = geomtest3 but wire sphere big
            +  1905/15831183/15855017: n1217
            +  **Doesn't blow up.** So it is the gradient penalty piece that makes it go wild. 
            +  Interpolation looks quite natural though.
            +  But still loss is a lot more spikey than when we normalise wire_sphere.
    +  15858166: n1222 = n1217 + branches go back further in G, no common branch, geomtest3
    +  15858329: n1223 = n1217 + common branch is longer (short branches)
        +  Faster than ^
    +  15863865: n1224 = n1217 + occupancy
        +  Suddenly less stable. Gen cannot compete it seems.
    +  
    +  Okay. So we can't disturb the equilibrium too much with the occupancy trick. But we can change the architecture quite a bit without impacting the apparent quality.
    +  I think there's no way for us to improve this model design, so we should just up the size or something.
    +  
    +  n1225 = n1223 + more G layers depending on seq_len
        + 15892131/15968116/16085991/16132278/16596380/16801078: sl = 512.
            + Slight overfit...
        + 15972362/16133246/16596447: sl=2048
    + dataset_augment6: 500k samples
        + 16865755/16956289/16985640: n1125, sl=512
        + n1226 = n1225 + **ABLATE** xy branch by not concatenating it into the input tensor.
            + 16896295/16956288/16985641: sl=512
        + n1227 = n1225 + **ABLATE** one-hot wire branch by not concatenating it into the input tensor.
            + 16896327/16956287/16985643: sl=512
