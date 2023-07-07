import utils
import wandb

"1) HPO"
for _ in range(5):
    confi = [0.014, 1, 75, "tanh", "Glorot normal", 1, 1, 1, 1, -1, 1]
    utils.inizia_hpo()
    output = utils.hpo(confi)
    # a = create_model(confi)
    # p = restore_model(a, confi)
    # utils.plot_3d(p, output)
    print("----------------------------------")
    utils.reset_iteration()

# utils.data_analysis("/home/giuglielmocappellini/Projects/PINNs/23.06.23_1D_system_no_source/output")

"2) Refinement: ricorda di modificare l'errore nella funzione test"
# confi = [0.001255, 4, 45, "tanh", "He uniform"]
# ini = 706

# endy = [1.5, 2, 5, 10, 20]
# weighty = [1, 10, 100, 1000, 10000, 100000, 1000000]

# for ef in endy:
#     for et in endy:
        
#         setts = [ef, et, 10, 1000, 100000, 1]
#             # start a new wandb run to track this script
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="refinement-no-source",

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

#         print(f"Inizio refinement {setts}")

#         utils.inizia_refinement(ini, confi, setts)
#         a = utils.create_model(confi, setts)
#         error = utils.train_model(a, confi, setts)

#         wandb.log({"err": error})
#         wandb.finish()

#         setts = [ef, et, 10, 10000, 100, 10000]
#             # start a new wandb run to track this script
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="refinement-no-source",

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

#         print(f"Inizio refinement {setts}")

#         utils.inizia_refinement(ini, confi, setts)
#         a = utils.create_model(confi, setts)
#         error = utils.train_model(a, confi, setts)

#         wandb.log({"err": error})
#         wandb.finish()


"3) Plot"
# utils.plot_3d(initial, confi, setts)
# utils.plot_err(initial, confi, setts)
