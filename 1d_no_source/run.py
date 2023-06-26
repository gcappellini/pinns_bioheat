import utils
import wandb

"1) HPO"
# setts = [1, 1, 1, 1, 1, 1]
# for _ in range(5):
#     confi = [0.014, 1, 75, "tanh", "Glorot normal"]
#     utils.inizia_hpo()
#     output = utils.hpo(confi)
#     a = create_model(confi, setts)
#     p = restore_model(a, confi, setts)
#     utils.plot_3d(p, output)
#     print("----------------------------------")
#     utils.reset_iteration()

# utils.data_analysis("/home/giuglielmocappellini/Projects/PINNs/23.06.23_1D_system_no_source/output")

"2) Refinement: ricorda di modificare l'output della funzione train"
# confi = [0.001255, 4, 45, "tanh", "He uniform"]
# ini = 706
#
# endy = [1.5, 2, 5, 10, 20]
# weighty = [1, 10, 100, 1000, 10000, 100000, 1000000]
#
# for ef in endy:
#     for et in endy:
#
#         setts = [ef, et, 10, 1000, 100000, 1]
#             # start a new wandb run to track this script
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="refinement-no-source",
#
#             # track hyperparameters and run metadata
#             config={
#                 "end_flux": ef,
#                 "end_time": et,
#                 "w_domain": 10,
#                 "w_bcl": 1000,
#                 "w_bcr": 100000,
#                 "w_ic": 1,
#             }
#         )
#
#         print(f"Inizio refinement {setts}")
#
#         utils.inizia_refinement(ini, confi, setts)
#         a = utils.create_model(confi, setts)
#         error = utils.train_model(a, confi, setts)
#
#         wandb.log({"err": error})
#         wandb.finish()
#
#         setts = [ef, et, 10, 10000, 100, 10000]
#             # start a new wandb run to track this script
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="refinement-no-source",
#
#             # track hyperparameters and run metadata
#             config={
#                 "end_flux": ef,
#                 "end_time": et,
#                 "w_domain": 10,
#                 "w_bcl": 10000,
#                 "w_bcr": 100,
#                 "w_ic": 10000,
#             }
#         )
#
#         print(f"Inizio refinement {setts}")
#
#         utils.inizia_esperimento(ini, confi, setts)
#         a = utils.create_model(confi, setts)
#         error = utils.train_model(a, confi, setts)
#
#         wandb.log({"err": error})
#         wandb.finish()


"3) Plot"
ini1 = 101
ini2 = 706

confi1 = [0.014, 1, 75, "tanh", "Glorot normal"]
confi2 = [0.001255, 4, 45, "tanh", "He uniform"]

setts1 = [1, 1, 1, 1, 1, 1]
setts2 = [10, 10, 10, 1000, 100000, 1]

utils.inizia_esperimento(ini1, confi1, setts1)
a1 = utils.create_model(confi1, setts1)
# p1 = utils.train_model(a1, confi1, setts1)
p1 = utils.restore_model(a1, confi1, setts1)

utils.inizia_esperimento(ini2, confi2, setts1)
a2 = utils.create_model(confi2, setts1)
# p2 = utils.train_model(a2, confi2, setts1)
p2 = utils.restore_model(a2, confi2, setts1)

utils.inizia_esperimento(ini2, confi2, setts2)
a3 = utils.create_model(confi2, setts2)
# p3 = utils.train_model(a3, confi2, setts2)
p3 = utils.restore2_model(a2, confi2, setts1)


utils.plot_3d(p1, confi1, setts1)
utils.plot_3d(p2, confi2, setts1)
utils.plot2_3d(p3, confi2, setts2)


