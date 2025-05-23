<!-- show different between guidance and method-->
<!DOCTYPE html class="has-navbar-fixed-top has-navbar-fixed-bottom">
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Dataset viewer </title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/1.0.2/css/bulma.min.css">
    <style>
      .image.is-128x256 {
          height: 128px;
          width: 256px;
      }
      .is-flip-image {
          transform: scaleX(-1);
      }
      body{
        width: 100vw;
      }
      
    </style>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@cityssm/bulma-sticky-table@3.0.0/bulma-with-sticky-table.min.css" />
  </head>
  <body>
    
    <div id="app" style="max-width: 100vw;">
      <section class="section" style="text-align: center;">
        <h1 class="title">Text Relighting: {{getDirection()}}</h1>
        <table class="table is-bordered mx-auto">
          <tbody>
            <tr>
              <td>
                Inversion prompt
              </td>
              <td>
                face of a boy
              </td>
            </tr>
            <tr>
              <td>
                Generation prompt
              </td>
              <td>
                face of a boy with sunlight illuminate on the left
              </td>
            </tr>
          </tbody>
        </table>
      </section>
        <table class="table is-bordered is-striped mx-auto" style="text-align: ;">
            <thead>
                <tr>
                    <th>
                        FileID
                    </th>
                    <th>
                      Source Image
                    </th>
                    <th>
                        DDIM
                    </th>
                    <th>
                        RF-Inversion (FluxDev)
                    </th>
                    <th>
                        RF-Inversion (SD3.5-Medium)
                    </th>
                </tr>
            </thead>
            <tbody>  
                <tr v-for="image_idx in image_index">
                  <th>{{image_idx}}</th>
                  <td>
                    <figure class="image is-128x128 mx-auto">
                      <img :src="'/output/datasets/face/face60k/images/'+getImageDir(image_idx)+'/'+getImageName(image_idx)+'.jpg'" loading="lazy"/>
                    </figure>
                  </td>
                  <td>
                    <figure class="image is-128x128 mx-auto">
                      <img :src="'/output/20241102/1.0/'+getDirection()+'/'+getImageName(image_idx)+'.jpg'" loading="lazy"/>
                    </figure>
                  </td>
                  <td>
                    <figure class="image is-128x128 mx-auto">
                      <img :src="'/output/20241102_fluxdev/'+getDirection()+'/'+getImageName(image_idx)+'.jpg'" loading="lazy"/>
                    </figure>
                  </td>
                  <td>
                    <figure class="image is-128x128 mx-auto">
                      <img :src="'/output/20241102_sd35med/'+getDirection()+'_6_15/'+getImageName(image_idx)+'.jpg'" loading="lazy"/>
                    </figure>
                  </td>
                </tr>
            </tbody>
        </table>
    </div>
        
  <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <script>
    async function getImageIndex(){
      const data = await response.json();
      return data;
    }
    const { createApp, ref } = Vue
      createApp({
        setup() {
          const image_index = ref([]);

          return {
            image_index,
          }
        },
        methods: {
          getImageName(file_id){
            let width = 5;
            return String(file_id).padStart(width, '0');
          },
          getImageDir(file_id){
            let width = 5;
            let dir_id = Math.floor(file_id / 1000) * 1000
            return String(dir_id).padStart(width, '0');
          },
          getDirection(){
            return "left";
          },
        },
        mounted() {
          var self = this;
          getImageIndex().then(ids =>{
            self.image_index = ids;
          });
        }
      }).mount('#app')
    </script>
  </body>
</html>