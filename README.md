<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/barnhemich/Sesame-Street">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Domain and Task Pretraining for Transformer Adapters</h3>

  <p align="center">
    Testing the the efficacy of using Domain and Task pretraining on Transformer adapters to improve performance of NLP tasks.
    <br />
    <a href="https://github.com/barnhemich/Sesame-Street/blob/main/Project_Report__Final.pdf"><strong>Read Our Findings »</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
When labeled training data for a specific NLP task is
limited, pretraining a language model on a large corpus
of unrelated data can provide significant performance improvements. After pretraining on the unrelated but abundant dataset, the model can then be fine-tuned using the
smaller task-specific dataset. Using the small labeled taskspecific dataset for additional pretraining has further increased model performance. Similarly, adding a pretraining step that uses unlabeled data from the same domain
as the task-specific data can provide benefits. However,
this process can be computationally expensive and inefficient due to the large number of parameters in the model.
It has been demonstrated that the parameter space can be
significantly decreased with minimal performance sacrifice
through the use of adapter modules for the final fine-tuning
step. In this study, we examine whether the benefits of domain adaptive-pretraining (DAPT) and task-adaptive pretraining (TAPT) can be combined with the use of adapter
modules to combine the efficiency of adapter modules with
the performance gains associated with TAPT and DAPT.
We compare performance of base RoBERTa, RoBERTa with
TAPT, RoBERTa with DAPT, and RoBERTa with both TAPT
and DAPT. We find that finetuning using adapters outperforms conventional finetuning on the full model, but
that TAPT and DAPT are less effective when pretraining
adapters only compared to when they are applied to the full
model.


<p align="right">(<a href="#readme-top">back to top</a>)</p>







<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/barnhemich/Sesame-Street.git
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@](https://twitter.com/) - email@gmail.com

Project Link: [https://github.com/barnhemich/Sesame-Street](https://github.com/barnhemich/Sesame-Street)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/barnhemich/Sesame-Street.svg?style=for-the-badge
[contributors-url]: https://github.com/barnhemich/Sesame-Street/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/barnhemich/Sesame-Street.svg?style=for-the-badge
[forks-url]: https://github.com/barnhemich/Sesame-Street/network/members
[stars-shield]: https://img.shields.io/github/stars/barnhemich/Sesame-Street.svg?style=for-the-badge
[stars-url]: https://github.com/barnhemich/Sesame-Street/stargazers
[issues-shield]: https://img.shields.io/github/issues/barnhemich/Sesame-Street.svg?style=for-the-badge
[issues-url]: https://github.com/barnhemich/Sesame-Street/issues
[license-shield]: https://img.shields.io/github/license/barnhemich/Sesame-Street.svg?style=for-the-badge
[license-url]: https://github.com/barnhemich/Sesame-Street/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/https://www.linkedin.com/in/michael-barnhart-973059171/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
